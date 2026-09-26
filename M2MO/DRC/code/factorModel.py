import numpy as np
from matplotlib import pyplot as plt
from DRC.code.utils import build_factor_correlation
import pandas as pd
from scipy import optimize


class FactorModel:
    """Calibration factorielle conforme à l'article :
    min_beta ||C0 - C(beta)||_F sous contrainte ||beta_i||^2 <= 1.
    """

    def __init__(self, returns, n_factors=None):
        self.returns = returns
        self.n_factors = n_factors
        self.tickers = list(returns.columns)
        self.n = len(self.tickers)
        self.T = len(returns)
        self.C0 = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.lambda_plus = None
        self.k_signal = None
        self.Beta_init = None
        self.Beta = None
        self.sigma_idio = None
        self.variance_explained = None
        self.factor_scores = None
        self.optim_result = None
        self.frobenius_error = None
        self.factor_corr_ = None

    def _standardize(self):
        return (self.returns - self.returns.mean()) / self.returns.std()

    def _pca_init(self, C0):
        vals, vecs = np.linalg.eigh(C0)
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_j = np.clip(vals[:self.n_factors], 0.0, None)
        vecs_j = vecs[:, :self.n_factors]
        beta0 = vecs_j @ np.diag(np.sqrt(vals_j))
        row_norm = np.linalg.norm(beta0, axis=1)
        beta0 = beta0 / np.maximum(1.0, row_norm)[:, None]
        return beta0, vals, vecs

    def _optimize_beta(self, C0, beta_init, maxiter=2000, tol=1e-10):
        n, K = beta_init.shape
        x0 = beta_init.ravel()

        def unpack(x):
            return x.reshape(n, K)

        def objective(x):
            beta = unpack(x)
            penalty = 1e4 * np.sum(np.clip(np.linalg.norm(beta, axis=1) - 1.0, 0.0, None)**2)
            Cb = build_factor_correlation(beta)
            return np.linalg.norm(C0 - Cb, ord='fro') + penalty

        constraints = [
            {'type': 'ineq', 'fun': (lambda x, i=i: 1.0 - np.sum(unpack(x)[i] ** 2))}
            for i in range(n)
        ]

        res = optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': tol, 'disp': False}
        )

        beta_hat = unpack(res.x)
        row_norm = np.linalg.norm(beta_hat, axis=1)
        beta_hat = beta_hat / np.maximum(1.0, row_norm)[:, None]
        return beta_hat, res
    
    def _get_factor_corr(self) -> np.ndarray:
        """
        Estimation empirique à partir des facteurs, identité sinon
        """
        self.factor_corr_ = np.corrcoef(self.factor_scores.values, rowvar=False)
        return self.factor_corr_

    def fit(self):
        R_std = self._standardize()
        R = R_std.values
        T, n = R.shape
        self.C0 = np.corrcoef(R, rowvar=False)
        q = n / T
        self.lambda_plus = (1 + np.sqrt(q)) ** 2
        evals, evecs = np.linalg.eigh(self.C0)
        idx = np.argsort(evals)[::-1]
        self.eigenvalues = evals[idx]
        self.eigenvectors = evecs[:, idx]
        n_signal = int((self.eigenvalues > self.lambda_plus).sum())
        self.k_signal = n_signal

        if self.n_factors is None:
            self.n_factors = max(n_signal, 2)

        self.Beta_init, _, _ = self._pca_init(self.C0)
        self.Beta, self.optim_result = self._optimize_beta(self.C0, self.Beta_init) # Minimisation de la fonction objectif
        sigma2 = 1 - (self.Beta ** 2).sum(axis=1)
        self.sigma_idio = np.sqrt(np.clip(sigma2, 1e-4, None))
        self.frobenius_error = np.linalg.norm(self.C0 - build_factor_correlation(self.Beta), ord='fro') # On peut rajouter le paramètre sigma_sys dans build_factor_correlation pour tenir compte de la correlation entre facteurs systémiques
        self.variance_explained = self.eigenvalues[:self.n_factors].sum() / self.eigenvalues.sum()
        V_K = self.eigenvectors[:, :self.n_factors]
        lam_K = np.clip(self.eigenvalues[:self.n_factors], 1e-12, None)
        self.factor_scores = pd.DataFrame(
            R @ V_K / np.sqrt(lam_K),
            index=self.returns.index,
            columns=[f'F{k+1}' for k in range(self.n_factors)]
        )

        self._get_factor_corr()

        print('FactorModelFrobenius calibre')
        print(f'  q = n/T = {q:.4f}')
        print(f'  lambda+ = {self.lambda_plus:.3f}')
        print(f'  Nb facteurs significatifs = {n_signal}')
        print(f'  K retenu = {self.n_factors}')
        print(f'  Variance expliquee = {self.variance_explained:.1%}')
        print(f'  Erreur de Frobenius = {self.frobenius_error:.4f}')
        return self

    def plot_factors(self, circle_axes=(2, 1), top_n=10, max_axes=6):
        """Visualise le cercle de correlation (cosinus directeurs ACP) et les contributeurs par axe.

        Parameters
        ----------
        circle_axes : tuple[int, int]
            Axes (1-indexes) utilises pour le cercle de correlation.
        top_n : int
            Nombre de tickers principaux affiches par axe.
        max_axes : int
            Nombre maximum d'axes factoriels affiches dans les contributeurs.
        """

        a1, a2 = circle_axes
        i, j = a1 - 1, a2 - 1
        
        # Calcul des cosinus directeurs (corrélations var-composantes)
        V_K = self.eigenvectors[:, :self.n_factors]
        lam_K = np.clip(self.eigenvalues[:self.n_factors], 1e-12, None)
        cosinus = V_K @ np.diag(np.sqrt(lam_K))  # shape: (n, K)
        
        x = cosinus[:, i]
        y = cosinus[:, j]

        fig, ax = plt.subplots(figsize=(9, 9))
        circle = plt.Circle((0, 0), 1.0, color='grey', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(circle)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        ax.scatter(x, y, alpha=0.75, color='#D04A02', edgecolor='black', linewidth=0.3)

        # Etiquette les points les plus informatifs pour eviter la surcharge visuelle
        label_strength = np.sqrt(x ** 2 + y ** 2)
        n_labels = min(top_n, len(self.tickers))
        label_idx = np.argsort(label_strength)[-n_labels:]
        for idx in label_idx:
            ax.text(x[idx], y[idx], self.tickers[idx], fontsize=9, ha='left', va='bottom')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel(f'Axe F{a1} ({self.eigenvalues[i]/self.eigenvalues.sum():.1%})')
        ax.set_ylabel(f'Axe F{a2} ({self.eigenvalues[j]/self.eigenvalues.sum():.1%})')
        ax.set_title('Cercle de correlation (cosinus directeurs ACP)')
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

        # 2) Principaux contributeurs par axe (utiliser aussi les cosinus pour cohérence)
        n_axes = min(self.n_factors, max_axes)
        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
        if n_axes == 1:
            axes = [axes]

        for k in range(n_axes):
            cosinus_k = cosinus[:, k]
            idx_top = np.argsort(np.abs(cosinus_k))[-min(top_n, len(cosinus_k)):]
            idx_top = idx_top[np.argsort(np.abs(cosinus_k[idx_top]))]
            tickers_top = [self.tickers[t] for t in idx_top]
            values_top = cosinus_k[idx_top]

            colors = ['#D04A02' if v >= 0 else '#7D7D7D' for v in values_top]
            axes[k].barh(tickers_top, values_top, color=colors, edgecolor='black', linewidth=0.3)
            axes[k].axvline(0, color='black', linewidth=0.8)
            var_exp = self.eigenvalues[k] / self.eigenvalues.sum()
            axes[k].set_title(f'Top {len(values_top)} contributeurs - F{k + 1} ({var_exp:.1%})')
            axes[k].set_xlabel('Cosinus directeur')

        fig.suptitle('Principaux contributeurs par axe factoriel (ACP)', y=1.02)
        plt.tight_layout()
        plt.show()

class FactorModelComparator:
    def __init__(self, model_initial, model_frobenius):
        self.model_initial = model_initial
        self.model_frobenius = model_frobenius

    def summary(self):
        C_init = build_factor_correlation(self.model_initial.Beta)
        C_frob = build_factor_correlation(self.model_frobenius.Beta)
        return pd.DataFrame({
            'Approche initiale': [
                np.linalg.norm(self.model_frobenius.C0 - C_init, ord='fro'),
                self.model_initial.K,
                self.model_initial.variance_explained,
                np.mean(self.model_initial.sigma_idio)
            ],
            'Approche Frobenius': [
                np.linalg.norm(self.model_frobenius.C0 - C_frob, ord='fro'),
                self.model_frobenius.K,
                self.model_frobenius.variance_explained,
                np.mean(self.model_frobenius.sigma_idio)
            ]
        }, index=['Erreur de Frobenius', 'K', 'Variance expliquee', 'Sigma idio moyen'])

    def beta_table(self):
        k = min(self.model_initial.K, self.model_frobenius.K)
        b0 = pd.DataFrame(self.model_initial.Beta[:, :k], index=self.model_initial.tickers,
                          columns=[f'beta_init_{j+1}' for j in range(k)])
        b1 = pd.DataFrame(self.model_frobenius.Beta[:, :k], index=self.model_frobenius.tickers,
                          columns=[f'beta_frob_{j+1}' for j in range(k)])
        return b0.join(b1, how='inner')
