"""
Fonctions utilitaires partagées entre les modèles de calibration et le
moteur DRC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def mean_std_standardizer(Z: np.ndarray) -> np.ndarray:
    mean_ = Z.mean(axis=0, keepdims=True)
    std_ = Z.std(axis=0, ddof=1, keepdims=True)
    std_ = np.where(std_ < 1e-12, 1.0, std_)
    return (Z - mean_) / std_


def build_factor_correlation(beta: np.ndarray, factor_corr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Reconstruit la matrice de corrélation implicite du modèle latent
    (chapitre 2, section 2.1.2 du mémoire) :

        C(beta, Sigma_Z) = beta @ Sigma_Z @ beta.T + D(beta),  diag(C) = 1

    Si `factor_corr` est une covariance (diagonale != 1, cas Sigma_Z^(0) =
    Lambda_J ou Sigma_Z_em avant renormalisation), le résultat reste bien
    une CORRÉLATION : seule la structure de dépendance entre facteurs
    importe pour la corrélation implicite entre actifs, l'échelle de
    Sigma_Z étant absorbée par beta.
    """
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


def compare_diagnostic_results(results_dict: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Synthétise les diagnostics statistiques (stationnarité, normalité)
    de plusieurs modèles calibrés, produits par `BaseFactorModel.diagnostics()`.

    Répond directement à la limite méthodologique identifiée au chapitre 1
    (section 1.5.5) : aucun des travaux recensés ne teste formellement ces
    hypothèses. Cette fonction en fait un contrôle systématique.
    """
    summary_rows = []
    for model_name, trained_model in results_dict.items():
        diag = trained_model.diagnostics()
        fs = diag["factor_stationarity"]
        fn = diag["factor_normality"]
        rmz = diag["residual_mean_zero"]
        rn = diag["residual_normality"]

        factor_df = fs.merge(fn, on="factor", how="outer")
        factor_df["stationarity_ok"] = (factor_df["ADF_pvalue"] < alpha) | (factor_df["KPSS_pvalue"] > alpha)
        factor_df["normality_ok"] = (factor_df["KS_pvalue"] > alpha) | (factor_df["Shapiro_pvalue"] > alpha)

        residual_df = rmz.merge(rn, on="asset", how="outer")
        residual_df["mean_zero_ok"] = residual_df["pvalue"] > alpha
        residual_df["normality_ok"] = (residual_df["KS_pvalue"] > alpha) | (residual_df["Shapiro_pvalue"] > alpha)

        summary_rows.append({
            "model": model_name,
            "n_factors": len(factor_df),
            "n_assets": len(residual_df),
            "factors_stationary_pct": float(factor_df["stationarity_ok"].mean()),
            "factors_normal_pct": float(factor_df["normality_ok"].mean()),
            "residuals_mean_zero_pct": float(residual_df["mean_zero_ok"].mean()),
            "residuals_normal_pct": float(residual_df["normality_ok"].mean()),
        })

    return pd.DataFrame(summary_rows)
