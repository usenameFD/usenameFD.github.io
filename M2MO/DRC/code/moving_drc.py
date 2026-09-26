"""
Comparaison des méthodes de calibration sur des fenêtres temporelles
(chapitre 3, section 3.3.4 du mémoire).

Ce module propose deux usages complémentaires :

1. `compute_annual_calibration_panel` -- panel de périodes ANNUELLES NON
   CHEVAUCHANTES (ex. 2008-01-01 à 2020-01-01 -> 12 périodes d'un an).
   Sur chaque période, pour chaque méthode de calibration (ACP itérative,
   Frobenius, EM, et éventuellement OLS/MLE en référence), on calibre le
   modèle latent sur les rendements de la période, puis on calcule :
     - l'écart entre corrélation implicite et corrélation empirique
       (norme de Frobenius, RMSE hors diagonale) ;
     - la corrélation moyenne implicite et la corrélation moyenne
       empirique (hors diagonale) ;
     - le DRC simulé pour le même portefeuille équipondéré.
   Le résultat est une table longue (une ligne par (période, méthode)),
   directement exploitable par `plot_metric_across_methods` /
   `plot_all_metrics` ci-dessous.

2. `compute_drc_rolling` -- fenêtre glissante à fréquence plus fine
   (hebdomadaire/mensuelle), conservée pour le suivi continu de la
   sensibilité d'UNE méthode donnée au choix de la fenêtre de calibration
   (chapitre 3, section 3.3.4, prolongement en continu de la comparaison
   ponctuelle sur les trois sous-périodes discrètes).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METHODS_DEFAULT = ["pca_iterative", "frobenius", "em"]


# ====================================================================
# VERSION 1 : panel annuel non chevauchant, multi-méthodes
# ====================================================================

def generate_annual_periods(
    start: str,
    end: str,
    period_years: float = 1.0,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Découpe l'intervalle [start, end] en périodes consécutives NON
    CHEVAUCHANTES de `period_years` années.

    Exemple : generate_annual_periods("2008-01-01", "2020-01-01") renvoie
    12 périodes ("2008-01-01","2009-01-01"), ..., ("2019-01-01","2020-01-01").

    La dernière période est tronquée à `end` si elle ne correspond pas à
    un multiple exact de `period_years` (elle est alors plus courte).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    offset = pd.DateOffset(months=int(round(period_years * 12)))

    periods = []
    cursor = start_ts
    while cursor < end_ts:
        period_end = min(cursor + offset, end_ts)
        periods.append((cursor, period_end))
        cursor = period_end

    return periods


def compute_annual_calibration_panel(
    tickers: Sequence[str],
    prices: pd.DataFrame,
    prob_default,
    lgd,
    factor_model_class,
    drc_model_class,
    n_factors: Optional[int],
    start: str,
    end: str,
    period_years: float = 1.0,
    methods: Optional[List[str]] = None,
    method_kwargs: Optional[Dict[str, dict]] = None,
    max_missing_pct: float = 0.05,
    M: int = 100_000,
    batch_size: int = 10_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Panel annuel non chevauchant : pour chaque période et chaque méthode
    de calibration, calibre le modèle latent, compare la corrélation
    implicite à la corrélation empirique, et simule le DRC du même
    portefeuille équipondéré.

    IMPORTANT (comparabilité inter-périodes) : l'univers de titres est
    déterminé UNE SEULE FOIS sur l'ensemble de la plage [start, end], et
    reste IDENTIQUE pour toutes les périodes. Un titre absent (ou trop
    lacunaire) sur ne serait-ce qu'une seule période de l'intervalle est
    exclu de TOUTES les périodes, plutôt que d'être conservé sur les
    périodes où il est disponible et exclu ailleurs. Cela garantit que
    les écarts observés entre périodes reflètent un changement de régime
    de marché, et non un changement de composition du portefeuille.

    Parameters
    ----------
    tickers : univers CANDIDAT initial, réduit à l'univers commun retenu
        sur l'ensemble de la plage [start, end] (voir `max_missing_pct`).
    prices : DataFrame de prix de clôture, indexé par date.
    prob_default, lgd : tableaux alignés sur `tickers` (probabilités de
        défaut et pertes en cas de défaut).
    n_factors : nombre de facteurs, commun à toutes les périodes et
        méthodes (cohérence de la comparaison, cf. chapitre 2, remarque
        sur Marchenko-Pastur). Si None, chaque période/méthode utilise
        son propre critère de sélection automatique.
    start, end : bornes de l'intervalle balayé (ex. "2008-01-01",
        "2020-01-01").
    period_years : durée de chaque période non chevauchante (1.0 par
        défaut, soit des périodes annuelles).
    methods : méthodes de calibration comparées (par défaut : ACP
        itérative, Frobenius, EM -- les trois méthodes structurantes du
        chapitre 2 ; OLS/MLE peuvent être ajoutées à titre de référence).
    method_kwargs : arguments spécifiques par méthode, ex.
        {"em": {"max_iter": 300, "update_factor_corr": True}}.
    max_missing_pct : seuil de tolérance aux données manquantes, appliqué
        UNE SEULE FOIS sur l'ensemble de la plage [start, end] pour fixer
        l'univers commun (et non période par période, contrairement à la
        version précédente).
    M, batch_size : paramètres de la simulation Monte Carlo du DRC.

    Returns
    -------
    pd.DataFrame (table longue), une ligne par (période, méthode), avec
    les colonnes :
        period_start, period_end, n_assets, method,
        frobenius_gap        : ||C0 - C(beta,Sigma_Z)||_F
        rmse_gap             : RMSE hors diagonale entre C0 et C_implicite
        mean_corr_empirical  : corrélation moyenne empirique (hors diag.)
        mean_corr_implied    : corrélation moyenne implicite (hors diag.)
        DRC_batch, DRC_error : DRC simulé et sa marge d'erreur (méthode
                                des batchs)
        EL, EL_error         : espérance de perte et sa marge d'erreur
        n_iter, converged    : diagnostics de convergence (si applicable)

    Note : `n_assets` est désormais constant sur toutes les lignes du
    panel (même univers pour toutes les périodes) ; la colonne est
    conservée pour transparence et vérification, mais ne varie plus.
    """
    from DRC.code.dataLoader import slice_period, equal_weighted_jtd

    methods = methods or METHODS_DEFAULT
    method_kwargs = method_kwargs or {}
    tickers = list(tickers)
    prob_default = np.asarray(prob_default, dtype=float)
    lgd = np.asarray(lgd, dtype=float)

    # ---- Univers commun fixé UNE SEULE FOIS sur toute la plage [start, end] ----
    full_window = slice_period(prices, tickers, start, end, max_missing_pct=max_missing_pct)
    common = [t for t in tickers if t in full_window.columns]
    dropped = [t for t in tickers if t not in common]
    if dropped and verbose:
        print(f"Univers fixé pour l'ensemble de la plage {start} -> {end} : "
              f"{len(common)}/{len(tickers)} titres retenus (exclus : {dropped})")

    idx_map = [tickers.index(t) for t in common]
    prob_default_common = prob_default[idx_map]
    lgd_common = lgd[idx_map]

    periods = generate_annual_periods(start, end, period_years=period_years)
    records = []

    for period_start, period_end in periods:
        period_label = f"{period_start.date()}"

        # On découpe désormais l'univers déjà FIXÉ, sans re-filtrer par période :
        # tout titre manquant ponctuellement au sein de sa propre fenêtre annuelle
        # (mais présent sur le reste de la plage) est simplement conservé via
        # ffill (déjà appliqué dans `full_window` par slice_period), pas exclu.
        sub_prices = full_window.loc[period_start:period_end, common]
        returns = sub_prices.pct_change().dropna()

        if len(returns) < 20:
            if verbose:
                print(f"[{period_label}] période ignorée (données insuffisantes : "
                      f"{len(returns)} observations)")
            continue

        jtd_period = equal_weighted_jtd(sub_prices, common, sub_prices.index[-1])

        for method in methods:
            kwargs = method_kwargs.get(method, {})
            model = factor_model_class(returns, n_factors=n_factors)
            model.fit()
            model.calibrate_beta(method, **kwargs)

            gap = model.compute_correlation_gap()

            drc_engine = drc_model_class(
                model, prob_default_common, lgd_common,
                (jtd_period[common] / jtd_period[common].sum()).values,
                M=M,
            )
            result = drc_engine.simulate(batch_size=batch_size, label=f"{period_label}_{method}")

            n_iter = model.metadata.get("pca_iterative_n_iter") or model.metadata.get("em_n_iter")
            converged = model.metadata.get("pca_iterative_converged")
            if converged is None:
                converged = model.metadata.get("em_converged")

            records.append({
                "period_start": period_start, "period_end": period_end,
                "period_label": period_label,
                "n_assets": len(common), "method": method,
                "frobenius_gap": gap["frobenius_norm"],
                "rmse_gap": gap["rmse_offdiag"],
                "mean_corr_empirical": gap["mean_emp_corr"],
                "mean_corr_implied": gap["mean_implied_corr"],
                "DRC_sec": result.DRC_sec, "DRC_error": result.Error_sec,
                "EL": result.EL, "EL_error": result.EL_error,
                "n_iter": n_iter, "converged": converged,
                "shrink_delta": model.delta
            })

            if verbose:
                print(f"[{period_label}] {method:<14s} | "
                      f"n_assets={len(common):3d} | "
                      f"Frobenius={gap['frobenius_norm']:.4f} | "
                      f"corr_moy(impl.)={gap['mean_implied_corr']:.3f} | "
                      f"DRC={result.DRC_sec:.4f}")

    return pd.DataFrame(records)


# ====================================================================
# VERSION 2 : graphiques comparant les méthodes, un par métrique
# ====================================================================

# (colonne du panel, libellé d'axe, titre, colonne d'erreur associée ou None)
DEFAULT_METRICS: List[Tuple[str, str, str, Optional[str]]] = [
    ("frobenius_gap", r"$\|C_0 - C(\hat\beta,\hat\Sigma_Z)\|_F$",
     "Écart de Frobenius à la corrélation empirique", None),
    ("mean_corr_implied", "Corrélation moyenne implicite (hors diagonale)",
     "Corrélation moyenne implicite, par méthode", None),
    ("DRC_sec", "DRC (méthode des sections)",
     "DRC simulé, par méthode", "DRC_error"),
]


def plot_metric_across_methods(
    panel: pd.DataFrame,
    metric_col: str,
    ylabel: str = "",
    title: str = "",
    error_col: Optional[str] = None,
    methods: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (11, 5),
    show_empirical_reference: bool = False,
) -> plt.Figure:
    """
    Trace, sur une figure unique, l'évolution d'une seule métrique au fil
    des périodes annuelles, avec une courbe par méthode de calibration.
    Chaque appel produit SA PROPRE figure : les échelles ne sont donc
    jamais partagées entre métriques de nature différente (corrélation,
    écart de Frobenius, DRC), conformément à la demande.

    Parameters
    ----------
    panel : sortie de `compute_annual_calibration_panel`.
    metric_col : nom de la colonne à tracer (ex. "frobenius_gap").
    error_col : nom de la colonne d'erreur associée (ex. "DRC_error"),
        tracée sous forme de bande d'incertitude si fournie.
    show_empirical_reference : si True, superpose une ligne pointillée
        représentant la corrélation empirique moyenne (identique pour
        toutes les méthodes, utile pour `mean_corr_implied`).
    """
    methods = methods or sorted(panel["method"].unique())
    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        sub = panel[panel["method"] == method].sort_values("period_start")
        if sub.empty:
            continue
        x = sub["period_start"]
        y = sub[metric_col]
        line, = ax.plot(x, y, marker="o", label=method)

        if error_col is not None and error_col in sub.columns:
            err = sub[error_col]
            ax.fill_between(x, y - err, y + err, alpha=0.15, color=line.get_color())

    if show_empirical_reference and "mean_corr_empirical" in panel.columns:
        ref = (
            panel[["period_start", "mean_corr_empirical"]]
            .drop_duplicates(subset="period_start")
            .sort_values("period_start")
        )
        ax.plot(ref["period_start"], ref["mean_corr_empirical"],
                linestyle="--", color="black", linewidth=1.5,
                label="Corrélation empirique (référence)")

    ax.set_xlabel("Début de période")
    ax.set_ylabel(ylabel or metric_col)
    ax.set_title(title or f"{metric_col} par méthode")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_all_metrics(
    panel: pd.DataFrame,
    metrics: Optional[List[Tuple[str, str, str, Optional[str]]]] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, plt.Figure]:
    """
    Produit une figure PAR TYPE DE MÉTRIQUE (écart de Frobenius,
    corrélation moyenne implicite, DRC simulé par défaut), chacune
    comparant toutes les méthodes de calibration sur l'ensemble des
    périodes annuelles. Renvoie un dictionnaire {nom_metrique: figure}.

    L'écart de Frobenius et le DRC ont des échelles typiquement très
    différentes de la corrélation moyenne (bornée dans [-1,1]) : les
    tracer sur des figures séparées, plutôt que sur des axes partagés
    ou des sous-graphiques contraints à la même échelle, est ce qui
    permet de lire correctement chacune d'entre elles.
    """
    metrics = metrics or DEFAULT_METRICS
    figures = {}
    for metric_col, ylabel, title, error_col in metrics:
        show_ref = metric_col == "mean_corr_implied"
        fig = plot_metric_across_methods(
            panel, metric_col, ylabel=ylabel, title=title,
            error_col=error_col, methods=methods,
            show_empirical_reference=show_ref,
        )
        figures[metric_col] = fig
    return figures


# ====================================================================
# Fenêtre glissante à fréquence fine (suivi continu, méthode unique)
# ====================================================================

def compute_drc_rolling(
    tickers,
    prices: pd.DataFrame,
    prob_default,
    lgd,
    jtd_fn,
    factor_model_class,
    drc_model_class,
    n_factors: int,
    calibration_method: str = "em",
    calibration_kwargs: dict | None = None,
    lookback_years: float = 1.0,
    calc_freq: str = "W-FRI",
    M: int = 50_000,
    fixed_model=None,
):
    """
    DRC en fenêtres glissantes à fréquence fine, pour UNE méthode de
    calibration donnée (contrairement au panel annuel ci-dessus, qui
    compare plusieurs méthodes sur des périodes non chevauchantes).
    Utile pour observer, jour après jour ou semaine après semaine, la
    trajectoire continue du DRC plutôt que sur des points annuels
    discrets (chapitre 3, section 3.3.4).

    Parameters
    ----------
    jtd_fn : callable(prices, tickers, date) -> pd.Series
    fixed_model : si fourni, la structure factorielle n'est PAS
        recalibrée à chaque date ; seules les expositions JTD le sont.
    """
    calibration_kwargs = calibration_kwargs or {}
    returns = prices[tickers].pct_change().dropna()
    lookback_days = int(round(lookback_years * 252))

    if len(returns) < lookback_days:
        raise ValueError(
            f"Historique insuffisant : {len(returns)} observations disponibles "
            f"pour un lookback de {lookback_days} jours."
        )

    calc_dates = returns.groupby(pd.Grouper(freq=calc_freq)).apply(lambda x: x.index.max()).dropna()
    min_valid_date = returns.index[lookback_days - 1]
    calc_dates = calc_dates[calc_dates >= min_valid_date]

    records = []
    for date in calc_dates:
        jtd = jtd_fn(prices, tickers, date)

        if fixed_model is not None:
            model = fixed_model
        else:
            sample = returns.loc[:date].tail(lookback_days)
            model = factor_model_class(sample, n_factors=n_factors)
            model.fit()
            model.calibrate_beta(calibration_method, **calibration_kwargs)

        drc_engine = drc_model_class(model, prob_default, lgd, jtd, M=M)
        result = drc_engine.simulate(label=f"{calibration_method}_{calc_freq}")

        records.append({
            "date": date,
            "DRC_sec": result.DRC_sec, "Error_sec": result.Error_sec,
            "DRC_batch": result.DRC_batch, "Error_batch": result.Error_batch,
            "EL": result.EL, "EL_error": result.EL_error,
        })

    return pd.DataFrame(records).set_index("date")


def plot_drc_timeseries_with_ci(drc_df: pd.DataFrame, figsize=(14, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    dates = drc_df.index
    drc_batch = 100 * drc_df["DRC_batch"].values
    error_batch = 100 * drc_df["Error_batch"].values

    ax.fill_between(dates, drc_batch - error_batch, drc_batch + error_batch,
                     alpha=0.3, color="steelblue", label="IC 95% (batch)")
    ax.plot(dates, drc_batch, "o-", color="steelblue", linewidth=2, markersize=5, label="DRC (batch)")
    ax.set_ylabel("DRC (%)")
    ax.set_xlabel("Date")
    ax.set_title("Trajectoire du DRC sur fenêtres glissantes")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig