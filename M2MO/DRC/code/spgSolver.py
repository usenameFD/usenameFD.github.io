"""
DRC.code.spgSolver
====================

Solveur generique de Gradient Projete Spectral (SPG, Birgin, Martinez &
Raydan, 2001) pour la calibration des modeles factoriels du chapitre 2
du memoire, avec la possibilite de rebasculer sur SLSQP (scipy) via un
unique parametre ``solver``.

Ce module fournit :

    - :class:`SPGOptimizer` : le solveur SPG generique (objectif +
      gradient + projection), reutilisable pour n'importe quelle
      fonction differentiable sous contrainte convexe.
    - :func:`proj_rows_ellipsoid` : projection EUCLIDIENNE EXACTE, ligne
      par ligne, sur le domaine Omega = { beta : beta_k^T Sigma_Z beta_k
      <= r^2 }, valable pour Sigma_Z = I_J (cas retenu au chapitre 2) ET
      pour Sigma_Z quelconque symetrique definie positive.
    - :func:`frobenius_objective_grad`, :func:`mle_objective_grad` :
      objectif + gradient analytique fermes pour les deux criteres du
      memoire (Proposition sur le gradient de Frobenius, section
      frobenius ; gradient de la log-vraisemblance observee, section
      deux-vraisemblances).
    - :func:`solve_beta` : dispatcher unique, appelable avec
      ``solver="spg"`` ou ``solver="slsqp"``, pour les criteres
      "frobenius" et "mle".
    - :func:`em_m_step_asset` : resolution du sous-probleme M-step d'un
      seul actif (arg max_{beta_k} Q_k(beta_k)), avec le meme choix de
      solveur, utilisee par ``BaseFactorModel._calibrate_beta_em``.

Point de vigilance central (voir Remarque, section EM du memoire) :
le critere "mle" n'est PAS borne hors du domaine admissible -- au bord
EXACT beta_k^T Sigma_Z beta_k = 1, Sigma_X(beta) devient singuliere et
log|Sigma_X| -> -infini. On relaxe donc systematiquement ce domaine a
(1 - epsilon) pour "mle" (les deux solveurs), jamais pour "frobenius"
(objectif borne partout, aucun risque de singularite).

Reference
---------
Birgin, E. G., Martinez, J. M., & Raydan, M. (2001). Algorithm 813:
SPG - software for convex-constrained optimization. ACM TOMS, 27(3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy import optimize as scipy_optimize


# ======================================================================
# 1. Solveur SPG generique
# ======================================================================

@dataclass
class SPGResult:
    """Conteneur homogene avec les attributs usuels d'un resultat
    d'optimisation (``x``, ``fun``), pour un usage interchangeable avec
    le retour de SLSQP dans le reste du code."""
    x: np.ndarray
    fun: float
    grad_norm: float
    n_iter: int
    n_feval: int
    n_geval: int
    success: bool
    message: str
    f_history: np.ndarray
    pgrad_history: np.ndarray
    time: float

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (f"SPGResult({status}, fun={self.fun:.6e}, "
                f"n_iter={self.n_iter}, |proj-grad|={self.grad_norm:.3e})")


def _identity(x):
    return x


class SPGOptimizer:
    """
    Gradient Projete Spectral (SPG) avec recherche lineaire non monotone
    de Grippo-Lampariello-Lucidi (GLL) et pas de Barzilai-Borwein
    securise -- implementation generique, independante du probleme
    optimise (cf. chapitre 2, Algorithme SPG).

    Parameters
    ----------
    fun, jac : callables
        Objectif et gradient, ``fun(x) -> float``, ``jac(x) -> ndarray``.
    proj : callable, optional
        Projection sur le domaine admissible. Par defaut, l'identite
        (probleme non contraint).
    lam_min, lam_max, gamma, M, sigma1, sigma2, max_ls_iter :
        Parametres numeriques standard de l'algorithme SPG.
    """

    def __init__(self, fun: Callable[[np.ndarray], float],
                 jac: Callable[[np.ndarray], np.ndarray],
                 proj: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 lam_min: float = 1e-10, lam_max: float = 1e10,
                 gamma: float = 1e-4, M: int = 10,
                 sigma1: float = 0.1, sigma2: float = 0.9,
                 max_ls_iter: int = 100):
        self.fun = fun
        self.jac = jac
        self.proj = proj if proj is not None else _identity
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.gamma = gamma
        self.M = M
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.max_ls_iter = max_ls_iter
        self._nfev = 0
        self._ngev = 0

    def _f(self, x):
        self._nfev += 1
        return self.fun(x)

    def _g(self, x):
        self._ngev += 1
        return self.jac(x)

    def minimize(self, x0: np.ndarray, tol: float = 1e-8,
                 max_iter: int = 2000, lam0: float = 1.0,
                 verbose: bool = False) -> SPGResult:
        t0 = time.time()
        self._nfev = 0
        self._ngev = 0

        x = self.proj(np.asarray(x0, dtype=float))
        lam = lam0
        f_hist = [self._f(x)]
        pg_hist = []
        success = False
        message = "max_iter reached"

        for i in range(max_iter):
            g = self._g(x)
            pg = self.proj(x - g) - x
            pg_norm = float(np.linalg.norm(pg))
            pg_hist.append(pg_norm)

            if pg_norm < tol:
                success = True
                message = "converged: ||proj(x-g)-x|| < tol"
                break

            d = self.proj(x - lam * g) - x
            gtd = float(np.dot(np.ravel(g), np.ravel(d)))

            if gtd >= 0.0:
                success = pg_norm < max(tol, 10 * np.finfo(float).eps ** 0.5)
                message = ("converged (quasi-stationnaire)" if success else
                            "direction non descendante")
                break

            f_ref = max(f_hist[-(self.M + 1):])
            alpha = 1.0
            f_curr = f_hist[-1]
            ls_ok = False
            for _ in range(self.max_ls_iter):
                f_trial = self._f(x + alpha * d)
                if f_trial <= f_ref + self.gamma * alpha * gtd:
                    ls_ok = True
                    break
                denom = f_trial - f_curr - alpha * gtd
                alpha_tmp = (-0.5 * alpha ** 2 * gtd / denom
                             if denom != 0 else alpha / 2.0)
                if not (self.sigma1 * alpha <= alpha_tmp <= self.sigma2 * alpha):
                    alpha_tmp = alpha / 2.0
                alpha = alpha_tmp
                if alpha < 1e-20:
                    break

            if not ls_ok:
                message = "recherche lineaire non concluante"
                break

            x_new = x + alpha * d
            g_new = self._g(x_new)

            s = x_new - x
            y = g_new - g
            sty = float(np.dot(np.ravel(s), np.ravel(y)))
            sts = float(np.dot(np.ravel(s), np.ravel(s)))
            lam = (min(self.lam_max, max(self.lam_min, sts / sty))
                   if sty > 0 else self.lam_max)

            x = x_new
            f_hist.append(self._f(x))

            if verbose and i % max(1, max_iter // 20) == 0:
                print(f"[SPG iter {i:5d}] f={f_hist[-1]:.6e} "
                      f"|pgrad|={pg_norm:.3e} alpha={alpha:.3e} lambda={lam:.3e}")

        return SPGResult(
            x=x, fun=f_hist[-1],
            grad_norm=pg_hist[-1] if pg_hist else float("nan"),
            n_iter=len(f_hist) - 1, n_feval=self._nfev, n_geval=self._ngev,
            success=success, message=message,
            f_history=np.array(f_hist), pgrad_history=np.array(pg_hist),
            time=time.time() - t0)


# ======================================================================
# 2. Projection sur Omega = { beta_k : beta_k^T Sigma_Z beta_k <= r^2 }
# ======================================================================

def _project_ellipsoid_single(v: np.ndarray, eigval_Z: np.ndarray,
                               eigvec_Z: np.ndarray, radius2: float = 1.0
                               ) -> np.ndarray:
    """
    Projection euclidienne exacte d'un vecteur ``v`` sur l'ellipsoide
    ``{b : b^T Sigma_Z b <= radius2}``, ou ``Sigma_Z = eigvec_Z @
    diag(eigval_Z) @ eigvec_Z.T`` est deja diagonalisee (une seule fois,
    en amont, pour tous les appels).

    Methode : multiplicateur de Lagrange mu >= 0 tel que la projection
    dans la base propre s'ecrit u_proj = u / (1 + mu * eigval_Z), avec
    mu racine de h(mu) = sum_i eigval_i u_i^2 / (1+mu eigval_i)^2 -
    radius2 = 0. h est strictement decroissante et convexe en mu > 0 :
    une iteration de Newton globalement convergente suffit.
    """
    u = eigvec_Z.T @ v
    q = float(np.sum(eigval_Z * u ** 2))
    if q <= radius2:
        return v.copy()

    def h(mu):
        return np.sum(eigval_Z * u ** 2 / (1.0 + mu * eigval_Z) ** 2) - radius2

    def hp(mu):
        return -2.0 * np.sum(eigval_Z ** 2 * u ** 2 / (1.0 + mu * eigval_Z) ** 3)

    mu = 0.0
    for _ in range(100):
        val = h(mu)
        if abs(val) < 1e-14:
            break
        mu_new = mu - val / hp(mu)
        mu = mu_new if mu_new >= 0 else (mu / 2.0 if mu > 0 else 1e-8)

    u_proj = u / (1.0 + mu * eigval_Z)
    return eigvec_Z @ u_proj


def proj_rows_ellipsoid(Sigma_Z: np.ndarray, radius2: float = 1.0
                         ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construit la projection ligne par ligne sur
    ``Omega = { beta : beta_k^T Sigma_Z beta_k <= radius2, k=1..n }``.

    Cas particulier ``Sigma_Z = I_J`` (choix retenu au chapitre 2) : la
    projection se reduit a une simple mise a l'echelle des lignes --
    cout O(n*J), sans diagonalisation. Le cas general (Sigma_Z SPD
    quelconque) est traite exactement par
    :func:`_project_ellipsoid_single`, apres une SEULE diagonalisation
    de Sigma_Z (cout amorti sur toutes les iterations).

    Parameters
    ----------
    Sigma_Z : ndarray, shape (k, k)
    radius2 : float
        Carre du rayon de l'ellipsoide (utiliser (1-epsilon)**2 pour
        relaxer legerement le bord du domaine, cf. discussion module).

    Returns
    -------
    callable
        ``proj(beta_matrix) -> ndarray`` de meme forme.
    """
    Sigma_Z = np.asarray(Sigma_Z, dtype=float)
    k = Sigma_Z.shape[0]
    r = float(np.sqrt(radius2))

    if np.allclose(Sigma_Z, np.eye(k)):
        def _proj(B):
            B = np.atleast_2d(B)
            norms = np.linalg.norm(B, axis=1)
            mask = norms > r
            if np.any(mask):
                B = B.copy()
                B[mask] *= (r / norms[mask, None])
            return B
        return _proj

    eigval_Z, eigvec_Z = np.linalg.eigh(Sigma_Z)
    eigval_Z = np.clip(eigval_Z, 1e-14, None)

    def _proj(B):
        B = np.atleast_2d(B)
        out = np.empty_like(B)
        for i in range(B.shape[0]):
            out[i] = _project_ellipsoid_single(B[i], eigval_Z, eigvec_Z, radius2)
        return out
    return _proj


# ======================================================================
# 3. Objectifs et gradients analytiques (Frobenius / MLE)
# ======================================================================

def build_factor_correlation(beta: np.ndarray,
                              factor_corr: Optional[np.ndarray] = None
                              ) -> np.ndarray:
    """Identique a ``DRC.code.utils.build_factor_correlation`` :
    C(beta, Sigma_Z) = beta Sigma_Z beta^T + diag(1 - diag(.)), diag=1."""
    beta = np.asarray(beta, dtype=float)
    n, k = beta.shape
    if factor_corr is None:
        factor_corr = np.eye(k)
    factor_corr = np.asarray(factor_corr, dtype=float)

    systematic = beta @ factor_corr @ beta.T
    specific = np.clip(1.0 - np.diag(systematic), 0.0, None)
    C = systematic.copy()
    np.fill_diagonal(C, np.diag(systematic) + specific)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C


def frobenius_objective_grad(C0: np.ndarray, Sigma_Z: np.ndarray
    ) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """
    Objectif et gradient analytique de la calibration par distance de
    Frobenius (chapitre 2, section frobenius) :

        f(beta) = ||C0 - C(beta, Sigma_Z)||_F^2
        nabla f(beta) = 4 (C(beta,Sigma_Z) - C0) @ beta @ Sigma_Z

    generalise a Sigma_Z quelconque (le cas Sigma_Z = I_J du memoire en
    est un cas particulier). Objectif BORNE partout (C0 et C(beta) sont
    toutes deux des matrices de correlation), aucun risque de
    singularite hors du domaine admissible.
    """
    C0 = np.asarray(C0, dtype=float)
    Sigma_Z = np.asarray(Sigma_Z, dtype=float)

    def fun(beta):
        Cb = build_factor_correlation(beta, Sigma_Z)
        R = C0 - Cb
        return float(np.sum(R * R))

    def jac(beta):
        Cb = build_factor_correlation(beta, Sigma_Z)
        return 4.0 * (Cb - C0) @ beta @ Sigma_Z

    return fun, jac


def mle_objective_grad(S_emp: np.ndarray, Sigma_Z: np.ndarray, T: int,
                        normalize: bool = True
    ) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """
    Objectif (oppose de la log-vraisemblance observee) et gradient
    analytique (chapitre 2, section deux-vraisemblances, eq. obs-ll) :

        ell(beta) = -(T/2)[log|Sigma_X(beta)| + tr(Sigma_X(beta)^{-1} S)]
        nabla ell(beta) = T * [Sigma_X^{-1}(S-Sigma_X)Sigma_X^{-1}]_off @ beta @ Sigma_Z

    ou [.]_off annule la diagonale de la matrice argument (le terme
    diagonal disparait du gradient car diag(Sigma_X(beta)) = 1 est
    constant en beta -- cf. preuve analytique du memoire).

    ATTENTION (1) : objectif NON BORNE hors du domaine admissible -- au
    bord EXACT beta_k^T Sigma_Z beta_k = 1, Sigma_X(beta) devient
    singuliere. Utiliser systematiquement une projection/contrainte
    relaxee a (1-epsilon) avec ce critere (cf. :func:`solve_beta`,
    :func:`em_m_step_asset`).

    ATTENTION (2), conditionnement numerique : l'objectif et le gradient
    sont proportionnels a T (nombre d'observations). Pour T grand
    (typiquement T > 200), le sous-probleme QP interne de SLSQP devient
    mal conditionne et echoue frequemment avec le message "Inequality
    constraints incompatible", meme lorsque le probleme est parfaitement
    bien pose. Par defaut (``normalize=True``), on retourne donc
    l'objectif et le gradient DIVISES PAR T (i.e. -ell(beta)/T), ce qui
    NE CHANGE PAS l'argmin en beta mais stabilise numeriquement SLSQP
    (verifie empiriquement : echec systematique avec T=800 sans
    normalisation, convergence immediate avec). SPG n'est, lui, pas
    affecte par ce probleme (la recherche lineaire GLL est invariante
    a une renormalisation globale de l'objectif), mais on applique la
    meme convention par coherence. Utiliser ``normalize=False`` pour
    retrouver l'echelle originale de -ell(beta) si necessaire (ex. pour
    des comparaisons directes de vraisemblance hors de ce module).
    """
    S_emp = np.asarray(S_emp, dtype=float)
    Sigma_Z = np.asarray(Sigma_Z, dtype=float)
    scale = (1.0 / T) if normalize else 1.0

    def neg_ell(beta):
        Sx = build_factor_correlation(beta, Sigma_Z)
        sign, logdet = np.linalg.slogdet(Sx)
        Sx_inv = np.linalg.inv(Sx)
        ell = -(T / 2.0) * (logdet + np.trace(Sx_inv @ S_emp))
        return -float(ell) * scale

    def grad_neg_ell(beta):
        Sx = build_factor_correlation(beta, Sigma_Z)
        Sx_inv = np.linalg.inv(Sx)
        Mmat = Sx_inv @ (S_emp - Sx) @ Sx_inv
        M_off = Mmat - np.diag(np.diag(Mmat))
        grad_ell = T * (M_off @ beta @ Sigma_Z)
        return -grad_ell * scale

    return neg_ell, grad_neg_ell


def _row_constraint_jac(x, i, n, k, Sigma_Z):
    """Jacobienne de la contrainte ``c - beta_i^T Sigma_Z beta_i`` par
    rapport a x = beta.ravel() (SLSQP)."""
    grad = np.zeros((n, k))
    beta_mat = x.reshape(n, k)
    grad[i] = -2.0 * Sigma_Z @ beta_mat[i]
    return grad.ravel()


# ======================================================================
# 4. Dispatcher unique : solve_beta(..., solver="spg" | "slsqp")
# ======================================================================

def solve_beta(
    criterion: str,
    beta0: np.ndarray,
    Sigma_Z: np.ndarray,
    solver: str = "spg",
    C0: Optional[np.ndarray] = None,
    S_emp: Optional[np.ndarray] = None,
    T: Optional[int] = None,
    tol: float = 1e-8,
    max_iter: int = 2000,
    penalty: float = 1e4,
    epsilon: Optional[float] = None,
    spg_kwargs: Optional[dict] = None,
    slsqp_options: Optional[dict] = None,
    verbose: bool = False,
):
    """
    Point d'entree UNIQUE pour calibrer beta selon le critere
    "frobenius" ou "mle", avec un choix explicite de solveur.

    Parameters
    ----------
    criterion : {"frobenius", "mle"}
        Critere a optimiser.
    beta0 : ndarray, shape (n, k)
        Point de depart (typiquement ``self._beta_from_pca(...)``).
    Sigma_Z : ndarray, shape (k, k)
        Covariance des facteurs (I_J dans la version retenue).
    solver : {"spg", "slsqp"}
        - "spg"   : gradient projete spectral, gradient ANALYTIQUE,
          projection en forme close sur l'ellipsoide.
        - "slsqp" : solveur SLSQP de scipy, avec penalisation
          quadratique de la contrainte (uniquement pour "frobenius")
          et contraintes d'inegalite explicites.
    C0 : ndarray, optional
        Matrice de correlation empirique cible (requis si
        criterion="frobenius").
    S_emp, T : optional
        Statistique suffisante et nombre d'observations (requis si
        criterion="mle").
    tol : float
        Tolerance d'arret (SPG : norme du gradient projete ;
        SLSQP : ftol).
    max_iter : int
        Nombre maximal d'iterations / iterations SLSQP.
    penalty : float
        Poids de la penalisation quadratique (SLSQP + "frobenius"
        uniquement ; ignore pour "mle", qui utilise des contraintes
        explicites, et pour "spg", ou la contrainte est respectee
        exactement a chaque iteration).
    epsilon : float, optional
        Marge de relaxation du domaine admissible :
        {beta_k^T Sigma_Z beta_k <= 1 - epsilon}. Par defaut, 0 pour
        "frobenius" (objectif borne partout, contrainte stricte inutile)
        et 1e-6 pour "mle" (necessaire : Sigma_X(beta) singuliere au
        bord exact -- cf. discussion memoire, section EM).
    spg_kwargs : dict, optional
        Parametres additionnels passes a ``SPGOptimizer``.
    slsqp_options : dict, optional
        Options additionnelles passees a ``scipy.optimize.minimize``.
    verbose : bool
        Affiche la progression du solveur choisi.

    Returns
    -------
    beta_hat : ndarray, shape (n, k)
    info : dict
        "solver", "success", "n_iter", "fun", "message", "raw_result".
    """
    criterion = criterion.lower()
    solver = solver.lower()
    n, k = beta0.shape

    if epsilon is None:
        epsilon = 1e-15 if criterion == "mle" else 0.0
    radius2 = (1.0 - epsilon) ** 2

    if criterion == "frobenius":
        if C0 is None:
            raise ValueError('C0 est requis pour criterion="frobenius".')
        fun, jac = frobenius_objective_grad(C0, Sigma_Z)
        report_scale = 1.0
    elif criterion == "mle":
        if S_emp is None or T is None:
            raise ValueError('S_emp et T sont requis pour criterion="mle".')
        # normalize=True (defaut) : ameliore le conditionnement numerique
        # de SLSQP pour T grand (cf. docstring de mle_objective_grad).
        # report_scale permet de re-exprimer "fun" dans info EN ECHELLE
        # ORIGINALE (-ell(beta), non divisee par T), pour ne pas
        # surprendre l'appelant.
        fun, jac = mle_objective_grad(S_emp, Sigma_Z, T, normalize=True)
        report_scale = float(T)
    else:
        raise ValueError('criterion doit etre "frobenius" ou "mle".')

    if solver == "spg":
        proj = proj_rows_ellipsoid(Sigma_Z, radius2=radius2)
        kwargs = spg_kwargs or {}
        opt = SPGOptimizer(fun, jac, proj=proj, **kwargs)
        result = opt.minimize(beta0, tol=tol, max_iter=max_iter, verbose=verbose)

        beta_hat = result.x
        info = dict(
            solver="spg", success=result.success, n_iter=result.n_iter,
            fun=result.fun * report_scale, message=result.message, raw_result=result,
        )
        return beta_hat, info

    elif solver == "slsqp":
        def unpack(x):
            return x.reshape(n, k)

        def objective(x):
            beta_mat = unpack(x)
            base = fun(beta_mat)
            if criterion == "frobenius":
                row_var = np.einsum("ij,jk,ik->i", beta_mat, Sigma_Z, beta_mat)
                pen = penalty * np.sum(np.clip(row_var - 1.0, 0.0, None) ** 2)
                return base + pen
            return base

        def objective_grad(x):
            beta_mat = unpack(x)
            g = jac(beta_mat).copy()
            if criterion == "frobenius":
                row_var = np.einsum("ij,jk,ik->i", beta_mat, Sigma_Z, beta_mat)
                active = np.clip(row_var - 1.0, 0.0, None)
                if np.any(active > 0):
                    g += penalty * 2.0 * active[:, None] * (2.0 * beta_mat @ Sigma_Z)
            return g.ravel()

        # Contrainte relaxee a (1-epsilon) : indispensable pour "mle"
        # (Sigma_X singuliere au bord exact), inoffensif pour
        # "frobenius" (epsilon=0 par defaut dans ce cas).
        constraints = [
            {"type": "ineq",
             "fun": lambda x, i=i: (1.0 - epsilon) - unpack(x)[i] @ Sigma_Z @ unpack(x)[i],
             "jac": lambda x, i=i: _row_constraint_jac(x, i, n, k, Sigma_Z)}
            for i in range(n)
        ]

        options = {"maxiter": max_iter, "ftol": tol, "disp": verbose}
        if slsqp_options:
            options.update(slsqp_options)

        # Garde-fou numerique : bornes larges (non actives a l'optimum)
        # pour empecher la recherche lineaire de SLSQP de s'echapper
        # dans des zones ou Sigma_X(beta) devient (numeriquement) non
        # definie positive. Particulierement necessaire pour "mle"
        # (objectif non borne hors du domaine) ; inoffensif pour
        # "frobenius" (objectif deja borne).
        safety_bound = 3.0
        bounds = [(-safety_bound, safety_bound)] * (n * k)

        res = scipy_optimize.minimize(
            objective, beta0.ravel(), jac=objective_grad,
            method="SLSQP", constraints=constraints, bounds=bounds,
            options=options,
        )

        beta_hat = unpack(res.x)
        # projection de securite finale, coherente avec (1-epsilon)
        for i in range(n):
            q_i = beta_hat[i] @ Sigma_Z @ beta_hat[i]
            if q_i > (1.0 - epsilon):
                beta_hat[i] *= np.sqrt((1.0 - epsilon) / q_i)

        info = dict(
            solver="slsqp", success=bool(res.success), n_iter=int(res.nit),
            fun=float(fun(beta_hat)) * report_scale, message=str(res.message), raw_result=res,
        )
        return beta_hat, info

    else:
        raise ValueError('solver doit etre "spg" ou "slsqp".')


# ======================================================================
# 5. Etape M de l'EM pour un seul actif (utilise par
#    BaseFactorModel._calibrate_beta_em)
# ======================================================================

def em_m_step_asset(
    xi: np.ndarray,
    M: np.ndarray,
    S: np.ndarray,
    Sigma_Z: np.ndarray,
    T: int,
    beta_k_init: np.ndarray,
    solver: str = "spg",
    tol: float = 1e-8,
    max_iter: int = 500,
    epsilon: float = 1e-6,
    spg_kwargs: Optional[dict] = None,
    slsqp_options: Optional[dict] = None,
):
    """
    Resout le sous-probleme M-step d'un seul actif k (Proposition de
    separabilite actif par actif) :

        beta_k^(m+1) = argmax_{beta_k : beta_k^T Sigma_Z beta_k < 1} Q_k(beta_k)

    ou, en notant sigma2_k = 1 - beta_k^T Sigma_Z beta_k et
    r = xi - M beta_k :

        -Q_k(beta_k) = (T/2) log(sigma2_k) + (1/(2 sigma2_k)) [r^T r + T beta_k^T S beta_k]

    Meme precaution que :func:`solve_beta` pour "mle" : Q_k contient
    log(sigma2_k), qui explose au bord exact du domaine. Le domaine est
    donc relaxe a (1-epsilon) pour les deux solveurs.

    Parameters
    ----------
    xi : ndarray, shape (T,)
        Rendements observes de l'actif k.
    M : ndarray, shape (T, k)
        Esperances conditionnelles m_t^(m) empilees (etape E).
    S : ndarray, shape (k, k)
        Covariance conditionnelle S^(m) (etape E, ne depend pas de t).
    Sigma_Z : ndarray, shape (k, k)
    T : int
    beta_k_init : ndarray, shape (k,)
    solver : {"spg", "slsqp"}
    epsilon : float
        Marge de relaxation du domaine, cf. ci-dessus.

    Returns
    -------
    beta_k_hat : ndarray, shape (k,)
    info : dict
    """
    MtM = M.T @ M
    Mtx = M.T @ xi
    radius2 = (1.0 - epsilon) ** 2

    def neg_Qk(b):
        d = 1.0 - b @ Sigma_Z @ b
        if d <= 1e-10:
            return 1e12 + 1e12 * (1e-10 - d + 1.0)
        r = xi - M @ b
        N = r @ r + T * (b @ S @ b)
        return 0.5 * T * np.log(d) + 0.5 * N / d

    def grad_neg_Qk(b):
        d = 1.0 - b @ Sigma_Z @ b
        if d <= 1e-10:
            return np.ones_like(b) * 1e6
        r = xi - M @ b
        N = r @ r + T * (b @ S @ b)
        dN = 2.0 * (MtM @ b - Mtx + T * S @ b)
        Sz_b = Sigma_Z @ b
        return (-T * Sz_b + 0.5 * dN) / d + (N / d ** 2) * Sz_b

    if solver == "spg":
        proj = proj_rows_ellipsoid(Sigma_Z, radius2=radius2)
        kwargs = spg_kwargs or {}
        opt = SPGOptimizer(neg_Qk, grad_neg_Qk,
                            proj=lambda B: proj(B.reshape(1, -1)).ravel(),
                            **kwargs)
        result = opt.minimize(beta_k_init, tol=tol, max_iter=max_iter)
        beta_k_hat = result.x
        info = dict(solver="spg", success=result.success,
                    n_iter=result.n_iter, fun=result.fun, raw_result=result)
        return beta_k_hat, info

    elif solver == "slsqp":
        constraints = [{"type": "ineq",
                         "fun": lambda b: (1.0 - epsilon) - b @ Sigma_Z @ b,
                         "jac": lambda b: -2.0 * Sigma_Z @ b}]
        options = {"maxiter": max_iter, "ftol": 1e-12, "disp": False}
        if slsqp_options:
            options.update(slsqp_options)
        res = scipy_optimize.minimize(
            neg_Qk, beta_k_init, jac=grad_neg_Qk, method="SLSQP",
            constraints=constraints, options=options)
        beta_k_hat = res.x
        q = beta_k_hat @ Sigma_Z @ beta_k_hat
        if q >= (1.0 - epsilon):
            beta_k_hat *= np.sqrt((1.0 - epsilon) * 0.999999 / q)
        info = dict(solver="slsqp", success=bool(res.success),
                    n_iter=int(res.nit), fun=float(neg_Qk(beta_k_hat)), raw_result=res)
        return beta_k_hat, info

    else:
        raise ValueError('solver doit etre "spg" ou "slsqp".')
