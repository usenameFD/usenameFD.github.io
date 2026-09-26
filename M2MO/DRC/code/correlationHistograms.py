"""
Histogrammes de corrélations par paire, ventilés Financial / Non-Financial
(chapitre 3, section 3.2.5 du mémoire).

Ces fonctions consomment directement `comparator.results`
(MultiPeriodCalibrationComparator) pour comparer, sur une même figure,
la distribution des corrélations implicites obtenues par les trois
méthodes de calibration, sur chacune des sous-périodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from DRC.code.utils import build_factor_correlation


def pairwise_correlations(corr_matrix: np.ndarray, tickers: List[str],
                            group: Optional[Dict[str, str]] = None,
                            group_filter: Optional[str] = None) -> np.ndarray:
    """
    Extrait les corrélations hors-diagonale (triangle supérieur) d'une
    matrice de corrélation, avec filtrage optionnel par groupe sectoriel
    (ex. "Financial", "Non-Financial").

    Si `group_filter` est fourni, ne garde que les paires (i, j) dont
    LES DEUX actifs appartiennent à ce groupe.
    """
    n = corr_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    values = corr_matrix[iu]

    if group_filter is None or group is None:
        return values

    mask = np.array([
        group.get(tickers[i]) == group_filter and group.get(tickers[j]) == group_filter
        for i, j in zip(*iu)
    ])
    return values[mask]


def plot_correlation_histograms(comparator, period: str,
                                  methods: Optional[List[str]] = None, bins: int = 20,
                                  figsize=(15, 4)):
    """
    Pour une sous-période donnée, trace un histogramme des corrélations
    par paire pour chacune des méthodes de calibration, avec ventilation
    Financial / Non-Financial si `group` est fourni.
    """
    methods = methods # or ["pca_iterative", "frobenius", "em"]
    fig, axes = plt.subplots(1, len(methods), figsize=figsize, sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        model = comparator.get_model(period, method)
        C0 = model.C0
        C = build_factor_correlation(model.beta, model.factor_corr_)
        tickers = model.tickers

        emp_corr = pairwise_correlations(C0, tickers)
        imp_corr = pairwise_correlations(C, tickers)
        ax.hist(emp_corr, bins=bins, alpha=0.6, label="Empirical", color="lightblue")
        ax.hist(imp_corr, bins=bins, alpha=0.6, label="Implicit", color="darkorange")
        ax.legend(fontsize=8)

        ax.set_title(f"{method}\n({period})")
        ax.set_xlabel("Corrélation implicite")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Fréquence")
    plt.tight_layout()
    return fig


def correlation_summary_table(comparator, group: Optional[Dict[str, str]] = None,
                                methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Tableau récapitulatif des corrélations moyennes par paire, par
    méthode et par période, avec ventilation Financial / Non-Financial
    si `group` est fourni. Écho du Tableau 2 de Laurent et al. (2016),
    colonnes "Average correlation" / "Financial" / "Non-Financial".
    """
    methods = methods or ["pca_iterative", "frobenius", "em"]
    rows = []

    for period in comparator.returns_by_period:
        for method in methods:
            model = comparator.get_model(period, method)
            C = build_factor_correlation(model.beta, model.factor_corr_)
            tickers = model.tickers

            row = {
                "période": period,
                "méthode": method,
                "corr_moyenne": float(np.mean(pairwise_correlations(C, tickers))),
            }
            if group is not None:
                row["corr_moyenne_financial"] = float(np.mean(pairwise_correlations(C, tickers, group, "Financial")))
                row["corr_moyenne_non_financial"] = float(np.mean(pairwise_correlations(C, tickers, group, "Non-Financial")))
            rows.append(row)

        row = {
            "période": period,
            "méthode": "unconstrained",
            "corr_moyenne": float(np.mean(pairwise_correlations(model.C0, tickers))),
        }
        if group is not None:
            row["corr_moyenne_financial"] = float(np.mean(pairwise_correlations(model.C0, tickers, group, "Financial")))
            row["corr_moyenne_non_financial"] = float(np.mean(pairwise_correlations(model.C0, tickers, group, "Non-Financial")))
        rows.append(row)
            

    return pd.DataFrame(rows)
