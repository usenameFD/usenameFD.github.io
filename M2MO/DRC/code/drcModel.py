from dataclasses import dataclass
import numpy as np
import scipy.stats as stats
from tqdm import tqdm


@dataclass
class DRCResult:
    losses: np.ndarray
    weights: np.ndarray
    DRC_sec: float
    DRC_batch: float
    Error_sec: float
    Error_batch: float
    EL: float
    EL_error: float
    label: str = "central"


class DrcModel:
    """Moteur Monte-Carlo pour le DRC.

    Modèle latent :
        X_i = beta_i' F + sigma_i eps_i

    avec :
        F ~ N(0, Sigma_F)
        eps_i ~ N(0,1) iid

    Défaut :
        1{X_i < c_i}, c_i = Phi^-1(PD_i)

    Perte :
        L = sum_i 1{X_i < c_i} * JTD_i

    DRC :
        VaR_{99,9}(L)
    """

    def __init__(self, factor_model, PD, LGD, JTD, M=10_000, CI=0.95, seed=42):
        self.fm = factor_model
        self.PD = np.asarray(PD, dtype=float)
        self.LGD = np.asarray(LGD, dtype=float)
        self.JTD = np.asarray(JTD, dtype=float)
        self.M = int(M)
        self.seed = seed
        self.CI = CI

        self.n_factors = factor_model.n_factors if hasattr(factor_model, "n_factors") else factor_model.k_signal
        self.n = factor_model.n

        self.beta = np.asarray(
            factor_model.Beta if hasattr(factor_model, "Beta") else factor_model.beta,
            dtype=float
        )
        self.sigma_idio = np.asarray(factor_model.sigma_idio, dtype=float)

        self.c = stats.norm.ppf(self.PD)

        # Correlations des facteurs
        self.factor_cov = np.asarray(factor_model.factor_corr_, dtype=float)
        self.factor_cov_chol = np.linalg.cholesky(self.factor_cov)

    def simulate(self, batch_size=10_000, label="central"):

        rng = np.random.default_rng(self.seed)

        losses = np.zeros(self.M)
        weights = np.ones(self.M)
        jtd_times_lgd = self.JTD * self.LGD

        n_full_batches = self.M // batch_size
        remainder = self.M % batch_size

        batch_sizes = [batch_size] * n_full_batches
        if remainder > 0:
            batch_sizes.append(remainder)

        # CORRECTIF : dimensionner `drc` sur le nombre TOTAL de batchs
        # (n_batches), y compris le batch reliquat, et non sur
        # n_full_batches seul. Avec l'ancienne version, si M n'était pas
        # un multiple exact de batch_size, `drc[i]` provoquait un
        # IndexError dès i == n_full_batches (le batch reliquat).
        n_batches = len(batch_sizes)
        start = 0
        drc = np.zeros(n_batches)
        for i, bs in enumerate(tqdm(batch_sizes, desc=f"MC [{label}]", leave=False)):
            end = start + bs
            idx = slice(start, end)

            # Facteurs systémiques corrélés : F ~ N(0, Sigma_F)
            Z = rng.standard_normal((bs, self.n_factors))
            F_sim = Z @ self.factor_cov_chol.T

            # Bruit idiosyncratique
            eps = rng.standard_normal((bs, self.n))

            # Variable latente
            X = F_sim @ self.beta.T + eps @ np.diag(self.sigma_idio)

            # Défauts et pertes
            D = (X < self.c).astype(np.int8)
            losses[idx] = D @ jtd_times_lgd
            drc[i] = np.quantile(losses[idx], 0.999)

            start = end

        drc_bar_sec = np.quantile(losses, 0.999)

        # CORRECTIF : degrés de liberté et taille d'échantillon basés sur
        # n_batches (nombre réel de batchs utilisés pour l'estimation de
        # la marge d'erreur), et non sur n_full_batches qui ignorait le
        # dernier batch partiel. max(..., 1) évite une division par zéro
        # dans le cas dégénéré à un seul batch.
        dof = max(n_batches - 1, 1)
        S2_sec = ((drc - drc_bar_sec) ** 2).sum() / dof
        S2_batch = ((drc - drc.mean()) ** 2).sum() / dof
        margin_sec = stats.t.ppf(1 - (1 - self.CI) / 2, dof) * np.sqrt(S2_sec / n_batches)
        margin_batch = stats.norm.ppf(1 - (1 - self.CI) / 2) * np.sqrt(S2_batch / n_batches)
        margin_el = stats.norm.ppf(1 - (1 - self.CI) / 2) * np.sqrt(losses.var() / self.M)

        return DRCResult(
            losses=losses,
            weights=weights,
            EL=float(losses.mean()),
            EL_error=float(margin_el),
            DRC_sec=float(np.quantile(losses, 0.999)),
            Error_sec=float(margin_sec),
            DRC_batch=float(drc.mean()),
            Error_batch=float(margin_batch),
            label=label
        )

    @staticmethod
    def report(result, JTD_total=None):
        print(f"\nRESULTATS DRC [{result.label}]")
        print("=" * 50)
        print(f"  E[L]          = {result.EL :.4f}")
        print(f"  DRC (sec)   = {result.DRC_sec :.4f}")
        print(f"  DRC (batch) = {result.DRC_batch :.4f}")
        print(f"  Intervalle de confiance (sec)   = [{result.DRC_sec - result.Error_sec :.4f}, {result.DRC_sec + result.Error_sec :.4f}]")
        print(f"  Intervalle de confiance (batch) = [{result.DRC_batch - result.Error_batch :.4f}, {result.DRC_batch + result.Error_batch :.4f}]")
