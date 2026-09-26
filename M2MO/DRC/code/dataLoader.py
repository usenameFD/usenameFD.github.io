"""
Chargement des données et construction du portefeuille équipondéré.

Conformément au chapitre 3 du mémoire (section 3.1), on abandonne la
réplication du CAC 40 par optimisation de Markowitz : le portefeuille
d'étude est directement composé des K constituants du CAC 40, chacun
affecté d'un poids w_k = 1/K. Cela évite de mélanger l'effet de la
méthode de calibration factorielle (l'objet de la comparaison) avec
l'effet, distinct, d'une construction de portefeuille par optimisation.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dateutil.utils import today


class DataLoader:
    """Télécharge et met en cache les prix de clôture ajustés (yfinance).

    Le cache est stocké une seule fois pour tout l'historique disponible ;
    chaque instance ne fait ensuite que découper la fenêtre [start, end]
    demandée. Pour les sous-périodes du mémoire (crise 2008, calme 2010s,
    Covid), utiliser de préférence `slice_period()` plutôt que de refaire
    un filtrage strict ici (voir docstring de `load_data`).
    """

    CACHE_PATH = "DRC/data/raw_prices.csv"

    def __init__(self, tickers, benchmark: str | None = None,
                 start: str = "2005-01-01", end: str | None = None):
        self.tickers = list(tickers)
        self.benchmark = benchmark
        self.start = start
        self.end = end
        self.prices: pd.DataFrame | None = None
        self.benchmark_prices: pd.Series | None = None

    def _all_tickers(self):
        return self.tickers + ([self.benchmark] if self.benchmark else [])

    def download(self):
        import yfinance as yf  # import paresseux : requis seulement au telechargement
        all_tickers = self._all_tickers()
        end = today().strftime("%Y-%m-%d")
        print(f"Téléchargement de {len(self.tickers)} constituants"
              f"{' + benchmark' if self.benchmark else ''}...")
        raw = yf.download(all_tickers, start=self.start, end=end,
                           auto_adjust=True, progress=False, group_by="ticker")

        if isinstance(raw.columns, pd.MultiIndex):
            data = pd.DataFrame({
                t: raw[t]["Close"] for t in all_tickers
                if t in raw.columns.get_level_values(0)
            })
        else:
            data = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        data.to_csv(self.CACHE_PATH)
        print(f"OK : {data.shape[0]} dates x {data.shape[1]} actifs")
        return self

    def load_data(self, start: str | None = None, end: str | None = None,
                  max_missing_pct: float = 0.5):
        """
        Charge les prix sur [start, end] et exclut les tickers dont le
        taux de valeurs manquantes SUR CETTE FENETRE dépasse
        `max_missing_pct`.

        CORRECTIF : l'ancienne version utilisait
        `dropna(axis=1, how="any")`, qui élimine une colonne au moindre
        NaN sur toute la période. Sur un historique de plusieurs années,
        un seul jour férié isolé (place boursière différente, glitch
        fournisseur) suffisait à exclure la quasi-totalité des tickers.

        Le seuil par défaut (0.5) est volontairement LARGE ici : ce
        chargement sert à constituer l'univers CANDIDAT sur tout
        l'historique. Le filtrage strict et pertinent doit se faire
        PAR SOUS-PERIODE via `slice_period()` (voir plus bas), pour ne
        pas exclure à tort, par exemple, un titre coté tardivement
        (Worldline, 2014) d'une sous-période où il est pourtant
        parfaitement disponible (ex. la période Covid, 2019-2021).
        """
        start = start or self.start
        end = end or self.end or today().strftime("%Y-%m-%d")

        if not os.path.exists(self.CACHE_PATH):
            self.download()

        data = pd.read_csv(self.CACHE_PATH, index_col=0, parse_dates=True)
        window = data.loc[start:end, self.tickers]

        missing_ratio = window.isna().mean()
        kept = missing_ratio[missing_ratio <= max_missing_pct].index.tolist()
        dropped = missing_ratio[missing_ratio > max_missing_pct].index.tolist()
        if dropped:
            detail = ", ".join(f"{t} ({missing_ratio[t]:.1%} manquant)" for t in dropped)
            print(f"Actifs exclus (> {max_missing_pct:.0%} de données manquantes sur "
                  f"l'historique complet) : {detail}")

        self.prices = window[kept].ffill()

        if self.benchmark:
            self.benchmark_prices = data.loc[start:end, self.benchmark].ffill()

        return self

    def get_returns(self, log: bool = False, time_horizon: int = 1):
        """Rendements sur `time_horizon` jours ouvrés (chap. 1, §137 : la
        fréquence des données de calibration se distingue de l'horizon
        d'un an retenu pour le calcul du DRC lui-même)."""
        if log:
            r = np.log(self.prices / self.prices.shift(time_horizon)).dropna()
        else:
            r = self.prices.pct_change(time_horizon).dropna()

        r_b = None
        if self.benchmark_prices is not None:
            if log:
                r_b = np.log(self.benchmark_prices / self.benchmark_prices.shift(time_horizon)).dropna()
            else:
                r_b = self.benchmark_prices.pct_change(time_horizon).dropna()

        return r, r_b


def slice_period(prices: pd.DataFrame, tickers, start: str, end: str,
                  max_missing_pct: float = 0.05) -> pd.DataFrame:
    """
    Découpe la fenêtre [start, end] d'un DataFrame de prix déjà chargé,
    et n'exclut un ticker que si son taux de valeurs manquantes SUR
    CETTE SEULE FENETRE dépasse `max_missing_pct` (5 % par défaut, plus
    strict que le chargement global -- voir `DataLoader.load_data`).

    C'est cette fonction, et non un filtrage sur l'historique complet,
    qui doit être utilisée pour découper les trois sous-périodes du
    mémoire (chapitre 3, section 3.1.2) : un titre correctement coté
    pendant la période Covid mais absent avant son IPO ne doit pas être
    pénalisé par des trous situés hors de la fenêtre considérée.
    """
    window = prices.loc[start:end, [t for t in tickers if t in prices.columns]]
    missing_ratio = window.isna().mean()
    kept = missing_ratio[missing_ratio <= max_missing_pct].index.tolist()
    dropped = missing_ratio[missing_ratio > max_missing_pct].index.tolist()
    if dropped:
        detail = ", ".join(f"{t} ({missing_ratio[t]:.1%} manquant)" for t in dropped)
        print(f"  Exclus sur cette période (> {max_missing_pct:.0%} manquant) : {detail}")
    return window[kept].ffill().dropna(how="any")


def equal_weighted_jtd(prices: pd.DataFrame, tickers, date, notional: float = 1.0) -> pd.Series:
    """Construit les expositions JTD d'un portefeuille équipondéré à la
    date `date` (chapitre 3, section 3.3.1 : JTD_k = w_k * V0 = V0 / K,
    indépendant de la période de calibration des facteurs)."""
    w = pd.Series(1.0 / len(tickers), index=tickers)
    jtd = w * notional
    return jtd / jtd.sum()
