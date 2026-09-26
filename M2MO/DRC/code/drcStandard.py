from dataclasses import dataclass
from typing import Dict, Union, Optional
import numpy as np
import pandas as pd


@dataclass
class DRCStandardResult:
    position_level: pd.DataFrame
    obligor_level: pd.DataFrame
    total_drc: float
    total_jtd: float


class DrcStandard:
    """
    Calcul DRC CRR3 - approche standard - non-securitisations - portefeuille equity long-only.

    Formule long-only :
        DRC = sum_b RW_b * NetJTD_b

    avec :
        NetJTD_b = sum des JTD longues par obligor b

    Entrées minimales :
        - JTD par obligor/position
        - rating par obligor
    """

    RISK_WEIGHTS = {
        "CQS1": 0.005,
        "CQS2": 0.03,
        "CQS3": 0.06,
        "CQS4": 0.15,
        "CQS5": 0.30,
        "CQS6": 0.50,
        "UNRATED": 0.15,
        "DEFAULT": 1.00,
    }

    RATING_TO_CQS = {
        "AAA": "CQS1", "AA+": "CQS1", "AA": "CQS1", "AA-": "CQS1",
        "A+": "CQS2", "A": "CQS2", "A-": "CQS2",
        "BBB+": "CQS3", "BBB": "CQS3", "BBB-": "CQS3",
        "BB+": "CQS4", "BB": "CQS4", "BB-": "CQS4",
        "B+": "CQS5", "B": "CQS5", "B-": "CQS5",
        "CCC+": "CQS6", "CCC": "CQS6", "CCC-": "CQS6", "CC": "CQS6", "C": "CQS6",
        "UNRATED": "UNRATED", "NR": "UNRATED",
        "DEFAULT": "DEFAULT", "D": "DEFAULT",
    }

    def __init__(
        self,
        rating_to_cqs: Optional[Dict[str, str]] = None,
        risk_weights: Optional[Dict[str, float]] = None,
        strict: bool = True
    ):
        self.rating_to_cqs = rating_to_cqs or self.RATING_TO_CQS.copy()
        self.risk_weights = risk_weights or self.RISK_WEIGHTS.copy()
        self.strict = strict

    def _to_series(self, x, name: str) -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.copy()
            s.name = name
            return s
        if isinstance(x, dict):
            return pd.Series(x, name=name)
        return pd.Series(x, name=name)

    def _normalize_rating(self, rating: str) -> str:
        if pd.isna(rating):
            return "UNRATED"
        return str(rating).strip().upper()

    def _map_rating_to_cqs(self, rating: str) -> str:
        rating_norm = self._normalize_rating(rating)
        cqs = self.rating_to_cqs.get(rating_norm)
        if cqs is None:
            if self.strict:
                raise ValueError(f"Rating non reconnu dans le mapping rating_to_cqs: {rating}")
            return "UNRATED"
        return cqs

    def _map_cqs_to_rw(self, cqs: str) -> float:
        if cqs not in self.risk_weights:
            raise ValueError(f"CQS non reconnu dans la table des risk weights: {cqs}")
        return float(self.risk_weights[cqs])

    def compute(
        self,
        jtd: Union[pd.Series, Dict[str, float]],
        ratings: Union[pd.Series, Dict[str, str]]
    ) -> DRCStandardResult:
        jtd_s = self._to_series(jtd, "jtd")
        ratings_s = self._to_series(ratings, "rating")

        obligors = sorted(set(jtd_s.index).union(set(ratings_s.index)))
        df = pd.DataFrame(index=obligors)
        df["jtd"] = jtd_s
        df["rating"] = ratings_s

        if df["jtd"].isna().any():
            missing = df.index[df["jtd"].isna()].tolist()
            raise ValueError(f"JTD manquant pour les obligors: {missing}")

        if df["rating"].isna().any():
            if self.strict:
                missing = df.index[df["rating"].isna()].tolist()
                raise ValueError(f"Rating manquant pour les obligors: {missing}")
            df["rating"] = df["rating"].fillna("UNRATED")

        if (df["jtd"] < 0).any():
            bad = df.index[df["jtd"] < 0].tolist()
            raise ValueError(f"Portefeuille non long-only: JTD négatif pour {bad}")

        df["rating_normalized"] = df["rating"].map(self._normalize_rating)
        df["cqs"] = df["rating_normalized"].map(self._map_rating_to_cqs)
        df["risk_weight"] = df["cqs"].map(self._map_cqs_to_rw)

        df["net_jtd"] = df["jtd"].astype(float)
        df["drc_contribution"] = df["net_jtd"] * df["risk_weight"]

        position_level = df.reset_index().rename(columns={"index": "obligor"})

        obligor_level = (
            position_level
            .groupby(["obligor", "rating_normalized", "cqs", "risk_weight"], as_index=False)
            .agg(
                net_jtd=("net_jtd", "sum"),
                drc_contribution=("drc_contribution", "sum")
            )
        )

        total_drc = float(obligor_level["drc_contribution"].sum())
        total_jtd = float(obligor_level["net_jtd"].sum())

        return DRCStandardResult(
            position_level=position_level,
            obligor_level=obligor_level,
            total_drc=total_drc,
            total_jtd=total_jtd
        )

    @staticmethod
    def report(result: DRCStandardResult) -> None:
        print("\nRESULTATS DRC STANDARD CRR3 [Equity Long-Only]")
        print("=" * 60)
        print(f"Nombre d'obligors  : {len(result.obligor_level)}")
        print(f"JTD total          : {result.total_jtd:,.2f}")
        print(f"DRC Standard total : {result.total_drc:,.2f}")
        if result.total_jtd > 0:
            print(f"DRC / JTD total    : {result.total_drc / result.total_jtd:>.2%}")
