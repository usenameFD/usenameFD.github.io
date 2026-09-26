from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.covariance import LedoitWolf


from DRC.code.utils import build_factor_correlation
from DRC.code.spgSolver import solve_beta, em_m_step_asset


class BaseFactorModel(ABC):
    def __init__(
        self,
        returns: pd.DataFrame,
        n_factors: Optional[int] = None,
        shrink: Optional[bool] = True,
        factor_selection_method: str = "mp",
        variance_threshold: float = 0.8,
        min_factors: int = 1,
        max_factors: Optional[int] = None
    ):
        self.returns = returns.copy()

        self.n_factors = n_factors
        self.calibration_method = None

        self.factor_selection_method = factor_selection_method.lower()
        self.variance_threshold = variance_threshold
        self.min_factors = min_factors
        self.max_factors = max_factors

        self.tickers = list(self.returns.columns)
        self.T, self.n = self.returns.shape

        self.scaler = StandardScaler(with_mean=True, with_std=True)

        self.returns_std: Optional[pd.DataFrame] = None
        self.C0: Optional[np.ndarray] = None
        self.shrink: Optional[bool] = shrink
        self.delta: Optional[float] = None

        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.lambda_plus: Optional[float] = None
        self.k_signal: Optional[int] = None

        self.factor_scores: Optional[pd.DataFrame] = None
        self.factor_corr_: Optional[np.ndarray] = None

        self.beta: Optional[np.ndarray] = None
        self.sigma_idio: Optional[np.ndarray] = None
        self.residuals: Optional[pd.DataFrame] = None
        self.fitted_corr: Optional[np.ndarray] = None
        self.frobenius_error: Optional[float] = None
        self.variance_explained: Optional[float] = None
        self.loglik: Optional[float] = None

        # --- Sorties spécifiques à la calibration EM ---
        self.em_factor_scores: Optional[pd.DataFrame] = None
        self.em_factor_posterior_cov: Optional[np.ndarray] = None
        self.em_factor_scores_cov_empirical: Optional[np.ndarray] = None

        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Pipeline de base (standardisation, spectre, sélection du nombre de facteurs)
    # ------------------------------------------------------------------ #
    def _standardize_returns(self) -> None:
        X = self.scaler.fit_transform(self.returns.values)
        self.returns_std = pd.DataFrame(
            X,
            index=self.returns.index,
            columns=self.returns.columns
        )
        if self.shrink is not None and self.shrink:
            lw = LedoitWolf().fit(self.returns_std.values)
            self.C0 = lw.covariance_
            self.delta = lw.shrinkage_
        else:
            self.C0 = self.returns_std.corr().values
            


    def _compute_spectrum(self) -> None:
        evals, evecs = np.linalg.eigh(self.C0)
        idx = np.argsort(evals)[::-1]

        self.eigenvalues = evals[idx]
        self.eigenvectors = evecs[:, idx]

        q = self.n / self.T
        self.lambda_plus = (1.0 + np.sqrt(q)) ** 2
        self.k_signal = int((self.eigenvalues > self.lambda_plus).sum())

        self.metadata["q"] = q
        self.metadata["lambda_plus"] = self.lambda_plus
        self.metadata["k_signal"] = self.k_signal

    def _select_number_of_factors(self) -> None:
        """
        Détermine automatiquement le nombre de facteurs si self.n_factors est None.

        Méthodes disponibles :
        - 'mp'       : seuil de Marchenko-Pastur
        - 'variance' : seuil de variance expliquée cumulée
        - 'fixed'    : n_factors doit alors être fourni
        """
        if self.n_factors is not None:
            k = int(self.n_factors)
        else:
            method = self.factor_selection_method

            if method == "mp":
                k = self.k_signal

            elif method == "variance":
                cumulative = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues)
                k = int(np.searchsorted(cumulative, self.variance_threshold) + 1)

            elif method == "fixed":
                raise ValueError(
                    "Si factor_selection_method='fixed', il faut fournir n_factors."
                )

            else:
                raise ValueError(
                    "factor_selection_method doit être parmi "
                    "['mp', 'variance', 'fixed']."
                )

        k = max(k, self.min_factors)

        if self.max_factors is not None:
            k = min(k, self.max_factors)

        k = min(k, self.n)

        self.n_factors = int(k)
        self.metadata["selected_n_factors"] = self.n_factors

    def _compute_variance_explained(self) -> None:
        self.variance_explained = float(
            np.sum(self.eigenvalues[:self.n_factors]) / np.sum(self.eigenvalues)
        )

    @abstractmethod
    def _extract_factors(self) -> None:
        """
        Doit renseigner self.factor_scores (DataFrame T x K), avec des
        scores STANDARDISÉS (variance unitaire par facteur) -- cf.
        PCAFactorModel._extract_factors, qui applique
        `mean_std_standardizer` après extraction PCA.
        Peut aussi renseigner self.variance_explained / metadata.
        """
        pass

    def _get_factor_corr(self) -> np.ndarray:
        """
        Sigma_Z fixée à l'identité (chapitre 2 du mémoire : choix retenu
        à l'image de Laurent, Sestier & Thomas 2016). Cela suppose que
        les facteurs extraits par `_extract_factors` sont STANDARDISÉS
        (variance unitaire), et non l'échelle brute Lambda_J des valeurs
        propres de C0.

        Ce choix élimine par construction tout problème de conditionnement
        de Sigma_Z dans les optimisations SLSQP des méthodes Frobenius,
        ACP itérative et EM : la contrainte beta_i' Sigma_F beta_i <= 1
        devient simplement ||beta_i||^2 <= 1, quel que soit le point de
        départ utilisé par ces méthodes.
        """
        self.factor_corr_ = np.eye(self.n_factors)
        return self.factor_corr_

    # ------------------------------------------------------------------ #
    # Initialisation commune par décomposition spectrale, utilisée comme
    # point de départ pour Frobenius, ACP itérative et EM
    # ------------------------------------------------------------------ #
    def _beta_from_pca(self, C: np.ndarray, k: int) -> np.ndarray:
        """
        Initialisation par décomposition spectrale de C, projetée dans le
        domaine admissible beta_i' Sigma_F beta_i < 1.

        CORRECTIF : la normalisation doit porter sur la VRAIE contrainte
        du modèle, beta_i' Sigma_F beta_i, et non sur ||beta_i||^2 seule.
        Ces deux quantités ne coïncident que si Sigma_F = I_J -- ce qui
        est le cas dans cette version (cf. _get_factor_corr), mais on
        garde l'appel explicite à Sigma_F ci-dessous par robustesse, au
        cas où Sigma_F cesserait un jour d'être l'identité : sans cela,
        le point de départ x0 peut violer la contrainte réelle d'un
        facteur supérieur à 10 sur des portefeuilles de taille réaliste,
        forçant l'optimiseur SLSQP à repartir d'un point infaisable
        (jusqu'à 15x plus d'itérations, et risque de non-convergence si
        le budget d'itérations est atteint avant correction).
        """
        Sigma_F = self._get_factor_corr()

        vals0, vecs0 = np.linalg.eigh(C)
        idx0 = np.argsort(vals0)[::-1]
        vals0, vecs0 = vals0[idx0], vecs0[:, idx0]
        vals0_j = np.clip(vals0[:k], 0.0, None)
        beta = vecs0[:, :k] @ np.diag(np.sqrt(vals0_j))

        for i in range(beta.shape[0]):
            q_i = beta[i] @ Sigma_F @ beta[i]
            if q_i >= 0.999:
                beta[i] *= np.sqrt(0.999 / q_i)

        return beta

    # ------------------------------------------------------------------ #
    # Méthode 1 : optimisation directe de la norme de Frobenius
    # ------------------------------------------------------------------ #
    def _calibrate_beta_frobenius(
        self,
        solver: str = "slsqp",
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Calibration des beta via :
            min || C0 - C(beta, Sigma_F) ||_F^2
        sous contrainte :
            beta_i' Sigma_F beta_i <= 1

        Sigma_F = I_J dans cette version (cf. _get_factor_corr).

        Parameters
        ----------
        solver : {"slsqp", "spg"}
            "slsqp" (par défaut, compatible avec le comportement
            historique) utilise scipy.optimize.minimize avec
            pénalisation quadratique de la contrainte.
            "spg" utilise le gradient projeté spectral (Algorithme SPG,
            chapitre 2), avec gradient ANALYTIQUE et projection exacte
            sur le domaine -- pas de pénalisation nécessaire.
        tol : float, optional
            Tolérance d'arrêt. Défaut : 1e-10 (slsqp) / 1e-9 (spg).
        max_iter : int, optional
            Nombre maximal d'itérations. Défaut : 2000.
        solver_kwargs : dict, optional
            Arguments additionnels transmis au solveur choisi (ex.
            ``{"gamma": 1e-4, "M": 10}`` pour spg, ou
            ``{"slsqp_options": {...}}`` pour slsqp).
        """
        Sigma_Z = self._get_factor_corr()
        n, k = self.n, self.n_factors

        beta0 = self._beta_from_pca(self.C0, k)

        default_tol = 1e-9 if solver == "spg" else 1e-10
        beta, info = solve_beta(
            criterion="frobenius",
            beta0=beta0,
            Sigma_Z=Sigma_Z,
            solver=solver,
            C0=self.C0,
            tol=tol if tol is not None else default_tol,
            max_iter=max_iter if max_iter is not None else 2000,
            **(solver_kwargs or {}),
        )

        sigma2 = 1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_Z, beta)
        sigma_idio = np.sqrt(np.clip(sigma2, 1e-10, None))

        F = self.factor_scores.values
        X = self.returns_std.values
        fitted = F @ beta.T
        residuals = X - fitted

        self.beta = beta
        self.sigma_idio = sigma_idio
        self.residuals = pd.DataFrame(
            residuals,
            index=self.returns.index,
            columns=self.tickers
        )
        self.fitted_corr = build_factor_correlation(beta, factor_corr=Sigma_Z)
        self.frobenius_error = float(np.linalg.norm(self.C0 - self.fitted_corr, ord="fro"))
        self.loglik = self._observed_loglik(beta, Sigma_Z, X)

        self.metadata["optim_result"] = info
        self.metadata["optim_solver"] = solver
        self.metadata["factor_corr"] = Sigma_Z

    # ------------------------------------------------------------------ #
    # Méthode 2 : approche itérative type Principal Axis Factoring
    # (Andersen, Sidenius & Basu 2003 / Laurent, Sestier & Thomas 2016)
    # ------------------------------------------------------------------ #
    def _calibrate_beta_pca_iterative(
        self,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        """
        Calibration itérative des beta par ACP répétée sur une matrice de
        corrélation "réduite" (Principal Axis Factoring). Ne fait appel à
        aucun solveur d'optimisation continue (SLSQP ou SPG) -- inchangée.
        """
        X = self.returns_std.values
        F = self.factor_scores.values
        Sigma_F = self._get_factor_corr()

        n, k = self.n, self.n_factors
        C0 = self.C0

        eigval_F, eigvec_F = np.linalg.eigh(Sigma_F)
        eigval_F = np.clip(eigval_F, 1e-12, None)
        Sigma_F_inv_sqrt = eigvec_F @ np.diag(1.0 / np.sqrt(eigval_F)) @ eigvec_F.T

        beta_tilde = self._beta_from_pca(C0, k)
        h2 = np.clip(np.sum(beta_tilde ** 2, axis=1), 0.0, 0.999)

        history: List[float] = []
        n_iter_done = 0

        for iteration in range(max_iter):
            C_reduced = C0.copy()
            np.fill_diagonal(C_reduced, h2)

            beta_tilde_new = self._beta_from_pca(C_reduced, k)
            h2_new = np.clip(np.sum(beta_tilde_new ** 2, axis=1), 0.0, 0.999)

            delta = float(np.max(np.abs(h2_new - h2)))
            history.append(delta)

            beta_tilde = beta_tilde_new
            h2 = h2_new
            n_iter_done = iteration + 1

            if delta < tol:
                break

        beta = beta_tilde @ Sigma_F_inv_sqrt

        for i in range(n):
            q_i = beta[i] @ Sigma_F @ beta[i]
            if q_i >= 1.0:
                beta[i] *= np.sqrt(0.999999 / q_i)

        sigma2 = 1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_F, beta)
        sigma_idio = np.sqrt(np.clip(sigma2, 1e-10, None))

        fitted = F @ beta.T
        residuals = X - fitted

        self.beta = beta
        self.sigma_idio = sigma_idio
        self.residuals = pd.DataFrame(
            residuals,
            index=self.returns.index,
            columns=self.tickers
        )
        self.fitted_corr = build_factor_correlation(beta, factor_corr=Sigma_F)
        self.frobenius_error = float(np.linalg.norm(self.C0 - self.fitted_corr, ord="fro"))
        self.loglik = self._observed_loglik(beta, Sigma_F, X)

        self.metadata["factor_corr"] = Sigma_F
        self.metadata["pca_iterative_n_iter"] = n_iter_done
        self.metadata["pca_iterative_converged"] = bool(history and history[-1] < tol)
        self.metadata["pca_iterative_history"] = history

    # ------------------------------------------------------------------ #
    # Méthode 3 : EM (vraisemblance marginale, Z réellement latent,
    # Sigma_Z = I_J fixée)
    # ------------------------------------------------------------------ #
    def _observed_loglik(self, beta: np.ndarray, Sigma_Z: np.ndarray, X: np.ndarray) -> float:
        """
        Log-vraisemblance observée (marginale en Z) :
            ell(theta) = -T/2 [ log|Sigma_X| + (1/T) sum_t X_t' Sigma_X^{-1} X_t ]
        """
        T = X.shape[0]
        K =  X.shape[1]
        d = 1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_Z, beta)
        Sigma_X = beta @ Sigma_Z @ beta.T + np.diag(d)

        _, logdet = np.linalg.slogdet(Sigma_X)
        Sigma_X_inv = np.linalg.inv(Sigma_X)
        quad = np.einsum("ti,ij,tj->t", X, Sigma_X_inv, X).mean()

        return float(-0.5 * T * (logdet + quad + K * np.log(2 * np.pi)))

    def _calibrate_beta_em(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        m_step_maxiter: int = 200,
        m_step_solver: str = "slsqp",
        m_step_solver_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Calibration des beta par algorithme EM maximisant la vraisemblance
        MARGINALE observée de X, en traitant Z comme réellement latent.
        Sigma_Z = I_J est fixée (chapitre 2 : choix retenu à l'image de
        Laurent, Sestier & Thomas 2016) et n'est jamais ré-estimée : seule
        beta est mise à jour à chaque itération.

        L'étape M (arg max_{beta_k} Q_k(beta_k), un sous-problème par
        actif, Proposition de séparabilité) est déléguée à
        ``spgSolver.em_m_step_asset``, avec un choix explicite de solveur :

        Parameters
        ----------
        m_step_solver : {"slsqp", "spg"}
            "slsqp" (par défaut, compatible avec le comportement
            historique) : solveur local SLSQP sur chaque sous-problème
            de dimension J.
            "spg" : gradient projeté spectral, gradient ANALYTIQUE de
            Q_k, projection exacte sur le domaine.
        m_step_solver_kwargs : dict, optional
            Arguments additionnels transmis au solveur de l'étape M.

        Garde-fou conservé, essentiel à la garantie théorique de
        monotonie démontrée au chapitre 2 (Théorème de monotonie) : on
        n'accepte la mise à jour beta_k^(m+1) que si elle ne dégrade pas
        Q_k (EM généralisé / GEM, Dempster, Laird & Rubin 1977,
        section 4) ; sinon on conserve la valeur précédente. Ce
        garde-fou s'applique IDENTIQUEMENT, quel que soit le solveur
        choisi pour l'étape M.

        Sorties stockées
        ----------------
        self.beta, self.sigma_idio, self.residuals, self.fitted_corr,
        self.frobenius_error : comme les autres méthodes de calibration.
        self.em_factor_scores : m_t = E[Z_t | X_t].
        self.em_factor_posterior_cov : S(theta_hat) = Var(Z_t | X_t).
        self.em_factor_scores_cov_empirical : covariance empirique des
            m_t, à TITRE DIAGNOSTIQUE UNIQUEMENT (sous-estime I_J,
            cf. Proposition d'atténuation, chapitre 2).
        """
        X = self.returns_std.values          # (T, n)
        T, n = X.shape
        k = self.n_factors

        Sigma_Z = self._get_factor_corr()    # = I_J, fixée
        beta = self._beta_from_pca(self.C0, k)

        loglik_history: List[float] = []
        converged = False
        n_iter_done = 0
        m_step_kwargs = m_step_solver_kwargs or {}

        def neg_Q_i_of(xi, M, S, Sigma_Z, T, b):
            d_i = 1.0 - b @ Sigma_Z @ b
            if d_i <= 1e-10:
                return 1e12 + 1e12 * (1e-10 - d_i + 1.0)
            r = xi - M @ b
            N_i = r @ r + T * (b @ S @ b)
            return 0.5 * T * np.log(d_i) + 0.5 * N_i / d_i

        for iteration in range(max_iter):
            # ---------------- E-step (vectorisé sur t) ----------------
            d = np.clip(1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_Z, beta), 1e-10, None)
            Sigma_X = beta @ Sigma_Z @ beta.T + np.diag(d)
            Sigma_X_inv = np.linalg.inv(Sigma_X)

            M = X @ Sigma_X_inv @ beta @ Sigma_Z          # (T, k)
            S = Sigma_Z - Sigma_Z @ beta.T @ Sigma_X_inv @ beta @ Sigma_Z   # (k, k)

            # ---------------- M-step : beta, actif par actif ----------------
            beta_new = np.zeros_like(beta)

            for i in range(n):
                xi = X[:, i]

                b, _info_i = em_m_step_asset(
                    xi, M, S, Sigma_Z, T, beta[i],
                    solver=m_step_solver,
                    tol=1e-9,
                    max_iter=m_step_maxiter,
                    **m_step_kwargs,
                )

                # Garde-fou de monotonie (cf. docstring), IDENTIQUE quel
                # que soit le solveur utilisé pour produire b.
                if neg_Q_i_of(xi, M, S, Sigma_Z, T, b) <= \
                   neg_Q_i_of(xi, M, S, Sigma_Z, T, beta[i]) + 1e-10:
                    beta_new[i] = b
                else:
                    beta_new[i] = beta[i]

            beta = beta_new

            # ---------------- Suivi de la vraisemblance observée ----------------
            ll = self._observed_loglik(beta, Sigma_Z, X)
            loglik_history.append(ll)
            n_iter_done = iteration + 1

            if verbose:
                delta_str = (
                    f"{ll - loglik_history[-2]:.3e}" if len(loglik_history) > 1 else "nan"
                )
                print(f"[EM] iter {n_iter_done:3d} | loglik={ll:.6f} | delta={delta_str}")

            if len(loglik_history) > 1 and abs(loglik_history[-1] - loglik_history[-2]) < tol:
                converged = True
                break

        # ---------------- Quantités finales ----------------
        d_final = np.clip(1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_Z, beta), 1e-10, None)
        Sigma_X_final = beta @ Sigma_Z @ beta.T + np.diag(d_final)
        Sigma_X_inv_final = np.linalg.inv(Sigma_X_final)

        M_final = X @ Sigma_X_inv_final @ beta @ Sigma_Z              # (T, k)
        S_final = Sigma_Z - Sigma_Z @ beta.T @ Sigma_X_inv_final @ beta @ Sigma_Z

        sigma_idio = np.sqrt(d_final)
        residuals = X - M_final @ beta.T

        self.beta = beta
        self.sigma_idio = sigma_idio
        self.residuals = pd.DataFrame(
            residuals,
            index=self.returns.index,
            columns=self.tickers
        )
        self.fitted_corr = build_factor_correlation(beta, factor_corr=Sigma_Z)
        self.frobenius_error = float(np.linalg.norm(self.C0 - self.fitted_corr, ord="fro"))
        self.loglik = loglik_history[-1] if loglik_history else np.nan

        factor_cols = [f"F{j+1}" for j in range(k)]
        self.em_factor_scores = pd.DataFrame(
            M_final, index=self.returns.index, columns=factor_cols
        )
        self.em_factor_posterior_cov = S_final.copy()
        self.em_factor_scores_cov_empirical = np.cov(M_final, rowvar=False)

        self.metadata["factor_corr"] = Sigma_Z
        self.metadata["em_loglik_history"] = loglik_history
        self.metadata["em_n_iter"] = n_iter_done
        self.metadata["em_converged"] = converged
        self.metadata["em_m_step_solver"] = m_step_solver
        self.metadata["loglik"] = loglik_history[-1] if loglik_history else np.nan
        self.metadata["neg_loglik"] = -loglik_history[-1] if loglik_history else np.nan

    def _calibrate_beta_ml(
        self,
        solver: str = "spg",
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        solver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Calibration des beta via :
            max ell(X;beta)     (vraisemblance observée, éq. obs-ll)
        sous contrainte :
            beta_i' Sigma_F beta_i <= 1

        Sigma_F = I_J dans cette version (cf. _get_factor_corr).

        Parameters
        ----------
        solver : {"slsqp", "spg"}
            "slsqp" (par défaut) : scipy.optimize.minimize, contraintes
            d'inégalité explicites + bornes de sécurité (nécessaires,
            car l'objectif MLE n'est PAS borné hors du domaine
            admissible -- contrairement à Frobenius -- ce qui peut faire
            diverger la recherche linéaire de SLSQP sans ce garde-fou).
            "spg" : gradient projeté spectral, gradient ANALYTIQUE,
            projection exacte -- ne quitte jamais le domaine admissible,
            donc intrinsèquement plus robuste sur ce critère.
        """
        X = self.returns_std.values
        Sigma_Z = self._get_factor_corr()
        n, k = self.n, self.n_factors

        beta0 = self._beta_from_pca(self.C0, k)
        T = X.shape[0]
        S_emp = (X.T @ X) / T

        default_tol = 1e-8 if solver == "spg" else 1e-8
        beta, info = solve_beta(
            criterion="mle",
            beta0=beta0,
            Sigma_Z=Sigma_Z,
            solver=solver,
            S_emp=S_emp,
            T=T,
            tol=tol if tol is not None else default_tol,
            max_iter=max_iter if max_iter is not None else 2000,
            **(solver_kwargs or {}),
        )

        sigma2 = 1.0 - np.einsum("ij,jk,ik->i", beta, Sigma_Z, beta)
        sigma_idio = np.sqrt(np.clip(sigma2, 1e-10, None))

        F = self.factor_scores.values
        fitted = F @ beta.T
        residuals = X - fitted

        self.beta = beta
        self.sigma_idio = sigma_idio
        self.residuals = pd.DataFrame(
            residuals,
            index=self.returns.index,
            columns=self.tickers
        )
        self.fitted_corr = build_factor_correlation(beta, factor_corr=Sigma_Z)
        self.frobenius_error = float(np.linalg.norm(self.C0 - self.fitted_corr, ord="fro"))
        self.loglik = self._observed_loglik(beta, Sigma_Z, X)

        self.metadata["optim_result"] = info
        self.metadata["optim_solver"] = solver
        self.metadata["factor_corr"] = Sigma_Z

    # ------------------------------------------------------------------ #
    # Dispatcher
    # ------------------------------------------------------------------ #
    def calibrate_beta(self, method: str, **kwargs) -> None:
        """
        Parameters
        ----------
        method : {"frobenius", "mle", "pca_iterative", "em"}
        **kwargs :
            Transmis à la méthode de calibration choisie. Notamment :
            - "frobenius", "mle" : solver="slsqp"|"spg", tol, max_iter,
              solver_kwargs.
            - "em" : m_step_solver="slsqp"|"spg", m_step_solver_kwargs,
              max_iter, tol, m_step_maxiter, verbose.
        """
        self.calibration_method = method.lower()

        if self.calibration_method == "frobenius":
            self._calibrate_beta_frobenius(**kwargs)

        elif self.calibration_method == "mle":
            self._calibrate_beta_ml(**kwargs)

        elif self.calibration_method in ("pca_iterative", "pca-iterative"):
            self._get_factor_corr()
            self._calibrate_beta_pca_iterative(**kwargs)

        elif self.calibration_method == "em":
            self._get_factor_corr()
            self._calibrate_beta_em(**kwargs)

        else:
            raise ValueError(
                "calibration_method doit être 'frobenius', 'pca_iterative', 'mle' ou 'em'."
            )

    # ------------------------------------------------------------------ #
    # Écart entre corrélation implicite (modèle) et corrélation empirique (C0)
    # ------------------------------------------------------------------ #
    def compute_correlation_gap(
        self,
        empirical_corr: Optional[np.ndarray] = None,
        implied_corr: Optional[np.ndarray] = None,
        off_diagonal_only: bool = True,
    ) -> Dict[str, float]:
        C_emp = empirical_corr if empirical_corr is not None else self.C0
        C_imp = implied_corr if implied_corr is not None else self.fitted_corr

        if C_emp is None or C_imp is None:
            raise ValueError(
                "Les matrices de corrélation empirique (C0) et implicite "
                "(fitted_corr) doivent être disponibles. Avez-vous appelé "
                "fit() puis calibrate_beta() ?"
            )

        n = C_emp.shape[0]

        if off_diagonal_only:
            mask = ~np.eye(n, dtype=bool)
            emp_vec = C_emp[mask]
            imp_vec = C_imp[mask]
        else:
            emp_vec = C_emp.ravel()
            imp_vec = C_imp.ravel()

        diff = emp_vec - imp_vec

        frobenius_norm = float(np.linalg.norm(C_emp - C_imp, ord="fro"))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        max_abs_dev = float(np.max(np.abs(diff)))

        if np.std(emp_vec) > 0 and np.std(imp_vec) > 0:
            r_pearson = float(np.corrcoef(emp_vec, imp_vec)[0, 1])
        else:
            r_pearson = np.nan

        return {
            "frobenius_norm": frobenius_norm,
            "rmse_offdiag": rmse,
            "mae_offdiag": mae,
            "max_abs_dev_offdiag": max_abs_dev,
            "mean_emp_corr": float(np.mean(emp_vec)),
            "mean_implied_corr": float(np.mean(imp_vec)),
            "pearson_emp_vs_implied": r_pearson,
        }

    def compare_calibration_methods(
        self,
        methods: Optional[List[str]] = None,
        n_repeats: int = 1,
        **kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        if methods is None:
            methods = ["frobenius", "pca_iterative", "em"]

        rows = []
        for m in methods:
            method_kwargs = kwargs.get(m, {})

            elapsed_list = []
            for _ in range(max(1, n_repeats)):
                t0 = time.perf_counter()
                self.calibrate_beta(m, **method_kwargs)
                elapsed_list.append(time.perf_counter() - t0)

            gap = self.compute_correlation_gap()
            gap["method"] = m
            gap["elapsed_seconds"] = float(np.median(elapsed_list))
            if n_repeats > 1:
                gap["elapsed_seconds_std"] = float(np.std(elapsed_list))

            gap["mean_sigma_idio"] = float(np.mean(self.sigma_idio))
            gap["max_row_systematic_var"] = float(
                np.max(np.einsum(
                    "ij,jk,ik->i", self.beta, self._get_factor_corr(), self.beta
                ))
            )

            if m == "pca_iterative":
                gap["n_iter"] = self.metadata.get("pca_iterative_n_iter", np.nan)
                gap["converged"] = self.metadata.get("pca_iterative_converged", np.nan)

            if m == "em":
                gap["n_iter"] = self.metadata.get("em_n_iter", np.nan)
                gap["converged"] = self.metadata.get("em_converged", np.nan)
            gap["loglik"] = self.loglik 

            rows.append(gap)

        df = pd.DataFrame(rows).set_index("method")

        cols_order = [
            "frobenius_norm", "loglik", "rmse_offdiag", "mae_offdiag", "max_abs_dev_offdiag",
            "pearson_emp_vs_implied", "mean_emp_corr", "mean_implied_corr",
            "mean_sigma_idio", "max_row_systematic_var", "elapsed_seconds",
        ]
        extra_cols = [c for c in df.columns if c not in cols_order]
        return df[cols_order + extra_cols]

    # ------------------------------------------------------------------ #
    # Pipeline principal
    # ------------------------------------------------------------------ #
    def fit(self):
        self._standardize_returns()
        self._compute_spectrum()
        self._select_number_of_factors()
        self._extract_factors()
        self._compute_variance_explained()
        return self

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def test_factor_stationarity(self) -> pd.DataFrame:
        rows = []
        for col in self.factor_scores.columns:
            s = self.factor_scores[col].dropna()
            adf_stat, adf_pvalue, *_ = adfuller(s, autolag="AIC")
            try:
                kpss_stat, kpss_pvalue, *_ = kpss(s, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_pvalue = np.nan, np.nan
            rows.append({
                "factor": col, "ADF_stat": adf_stat, "ADF_pvalue": adf_pvalue,
                "KPSS_stat": kpss_stat, "KPSS_pvalue": kpss_pvalue
            })
        return pd.DataFrame(rows)

    def test_factor_normality(self) -> pd.DataFrame:
        rows = []
        for col in self.factor_scores.columns:
            s = self.factor_scores[col].dropna()
            z = (s - s.mean()) / s.std(ddof=1)
            ks_stat, ks_pvalue = stats.kstest(z, "norm")
            shapiro_stat, shapiro_pvalue = stats.shapiro(s)
            rows.append({
                "factor": col, "KS_stat": ks_stat, "KS_pvalue": ks_pvalue,
                "Shapiro_stat": shapiro_stat, "Shapiro_pvalue": shapiro_pvalue
            })
        return pd.DataFrame(rows)

    def test_residual_mean_zero(self) -> pd.DataFrame:
        rows = []
        for col in self.residuals.columns:
            r = self.residuals[col].dropna()
            t_stat, pvalue = stats.ttest_1samp(r, 0.0)
            rows.append({"asset": col, "mean_residual": r.mean(), "t_stat": t_stat, "pvalue": pvalue})
        return pd.DataFrame(rows)

    def test_residual_normality(self) -> pd.DataFrame:
        rows = []
        for col in self.residuals.columns:
            r = self.residuals[col].dropna()
            z = (r - r.mean()) / r.std(ddof=1)
            ks_stat, ks_pvalue = stats.kstest(z, "norm")
            shapiro_stat, shapiro_pvalue = stats.shapiro(r)
            rows.append({
                "asset": col, "KS_stat": ks_stat, "KS_pvalue": ks_pvalue,
                "Shapiro_stat": shapiro_stat, "Shapiro_pvalue": shapiro_pvalue
            })
        return pd.DataFrame(rows)

    def diagnostics(self) -> Dict[str, pd.DataFrame]:
        return {
            "factor_stationarity": self.test_factor_stationarity(),
            "factor_normality": self.test_factor_normality(),
            "residual_mean_zero": self.test_residual_mean_zero(),
            "residual_normality": self.test_residual_normality()
        }

    def summary(self) -> pd.DataFrame:
        out = {
            "n_obs": self.T,
            "n_assets": self.n,
            "n_factors": self.n_factors,
            "factor_selection_method": self.factor_selection_method,
            "k_signal_mp": self.k_signal,
            "lambda_plus": self.lambda_plus,
            "variance_explained": self.variance_explained,
            "calibration_method": self.calibration_method,
            "frobenius_error": self.frobenius_error,
            "mean_sigma_idio": np.mean(self.sigma_idio) if self.sigma_idio is not None else np.nan
        }

        if self.calibration_method == "pca_iterative":
            out["pca_n_iter"] = self.metadata.get("pca_iterative_n_iter", np.nan)
            out["pca_converged"] = self.metadata.get("pca_iterative_converged", np.nan)

        if self.calibration_method == "em":
            out["neg_loglik"] = self.metadata.get("neg_loglik", np.nan)
            out["em_n_iter"] = self.metadata.get("em_n_iter", np.nan)
            out["em_converged"] = self.metadata.get("em_converged", np.nan)
            out["em_m_step_solver"] = self.metadata.get("em_m_step_solver", np.nan)
        if self.calibration_method in ("frobenius", "mle"):
            out["optim_solver"] = self.metadata.get("optim_solver", np.nan)
        out["loglik"] = self.loglik

        if self.factor_corr_ is not None:
            out["factor_corr_used"] = (
                "identity"
                if np.allclose(self.factor_corr_, np.eye(self.n_factors))
                else "custom_or_estimated"
            )

        return pd.DataFrame.from_dict(out, orient="index", columns=["value"])
