"""
Comparaison des trois méthodes de calibration (ACP itérative, Frobenius,
EM) sur plusieurs sous-périodes contrastées.

Implémente le protocole du chapitre 3 du mémoire, section 3.2
("Calibration du modèle latent") : pour chaque période et chaque méthode,
on exécute la calibration et on enregistre les diagnostics de convergence
et de qualité d'ajustement décrits en 3.2.1-3.2.5.

Le tableau produit par `summary_table()` est l'équivalent, pour ce
mémoire, du Tableau 2 de Laurent, Sestier & Thomas (2016) : une matrice
méthode x configuration avec l'écart de Frobenius comme métrique commune,
la différence étant que l'axe des colonnes correspond ici à des PÉRIODES
d'une même source de données plutôt qu'à des SOURCES différentes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


METHODS_DEFAULT = ["pca_iterative", "frobenius", "em"]


@dataclass
class PeriodCalibrationResult:
    period: str
    method: str
    model: object
    n_iter: Optional[int] = None
    converged: Optional[bool] = None
    frobenius_error: Optional[float] = None
    loglik: Optional[float] = None
    n_factors_used: int = 0
    n_factors_mp: Optional[int] = None
    heywood_count: int = 0
    elapsed_seconds: Optional[float] = None


class MultiPeriodCalibrationComparator:
    """
    Parameters
    ----------
    returns_by_period : dict[str, pd.DataFrame]
        Rendements par période, ex. {"crise_2008": df1, "calme_2010s": df2,
        "covid": df3}.
    factor_model_class : type
        Classe dérivée de BaseFactorModel (typiquement PCAFactorModel).
    n_factors : int, optional
        Nombre de facteurs commun aux trois périodes (chapitre 2,
        remarque "Nombre de facteurs et régime de marché") ; si None,
        chaque période utilise son propre critère de Marchenko-Pastur.
    heywood_eps : float
        Seuil en-deçà duquel une variance spécifique sigma_idio^2 est
        considérée comme un cas de Heywood (chapitre 2, Prop. Heywood).
    """

    def __init__(
        self,
        returns_by_period: Dict[str, pd.DataFrame],
        factor_model_class,
        shrink: Optional[bool] = None,
        n_factors: Optional[int] = None,
        heywood_eps: float = 1e-4,
        em_kwargs: Optional[dict] = None,
    ):
        self.returns_by_period = returns_by_period
        self.factor_model_class = factor_model_class
        self.n_factors = n_factors
        self.heywood_eps = heywood_eps
        self.em_kwargs = em_kwargs or {"max_iter": 200, "update_factor_corr": True}
        self.shrink = shrink

        self.results: List[PeriodCalibrationResult] = []
        self.models: Dict[str, object] = {}  # key: f"{period}__{method}"

    def run(self, methods: Optional[List[str]] = None, verbose: bool = True) -> "MultiPeriodCalibrationComparator":
        methods = methods or METHODS_DEFAULT
        self.results = []
        self.models = {}

        for period, returns in self.returns_by_period.items():
            for method in methods:
                model = self.factor_model_class(returns, n_factors=self.n_factors, shrink=self.shrink)
                model.fit()
                n_factors_mp = model.k_signal

                t0 = time.perf_counter()
                method_kwargs = self.em_kwargs if method == "em" else {}
                model.calibrate_beta(method, **method_kwargs)
                elapsed = time.perf_counter() - t0

                n_iter = None
                converged = None
                if method == "pca_iterative":
                    n_iter = model.metadata.get("pca_iterative_n_iter")
                    converged = model.metadata.get("pca_iterative_converged")
                elif method == "em":
                    n_iter = model.metadata.get("em_n_iter")
                    converged = model.metadata.get("em_converged")
                elif method == "frobenius" or method == "mle":
                    res = model.metadata.get("optim_result")
                    n_iter = getattr(res, "nit", None) if res is not None else None
                    converged = getattr(res, "success", None) if res is not None else None

                heywood_count = int(np.sum(model.sigma_idio ** 2 < self.heywood_eps))

                result = PeriodCalibrationResult(
                    period=period, method=method, model=model,
                    n_iter=n_iter, converged=converged,
                    frobenius_error=model.frobenius_error,
                    loglik=model.loglik,
                    n_factors_used=model.n_factors,
                    n_factors_mp=n_factors_mp,
                    heywood_count=heywood_count,
                    elapsed_seconds=elapsed,
                )
                self.results.append(result)
                self.models[f"{period}__{method}"] = model

                if verbose:
                    print(f"[{period:>14s}] {method:<14s} | n_iter={str(n_iter):>5s} | "
                          f"Frobenius={model.frobenius_error:.4f} | loglik={model.loglik:.4f}")

        return self

    def summary_table(self) -> pd.DataFrame:
        """Tableau méthode x période, écho du Tableau 2 de Laurent et al.
        (2016) : voir chapitre 3, section 3.2.5 du mémoire."""
        rows = []
        for r in self.results:
            rows.append({
                "période": r.period, "méthode": r.method,
                "J (retenu)": r.n_factors_used, "J (Marchenko-Pastur)": r.n_factors_mp,
                "n_iter": r.n_iter, "converged": r.converged,
                "cas_Heywood": r.heywood_count,
                "||C0 - C(beta,Sigma_Z)||_F": r.frobenius_error,
                "log-vraisemblance": r.loglik,
                "temps_calcul_s": r.elapsed_seconds,
            })
        df = pd.DataFrame(rows)
        return df.pivot(index="méthode", columns="période") if not df.empty else df

    def raw_table(self) -> pd.DataFrame:
        """Version non pivotée, plus simple à filtrer/exporter.

        CORRECTIF : `vars(r)` renvoie une RÉFÉRENCE DIRECTE à
        `r.__dict__`, pas une copie. Faire `.pop("model")` dessus
        supprimait donc irréversiblement l'attribut `model` de chaque
        objet `PeriodCalibrationResult` -- un simple appel à
        `raw_table()` cassait ensuite tout usage ultérieur de
        `comparator.results` (notamment dans `correlationHistograms.py`,
        qui a besoin de `r.model`). On copie explicitement le
        dictionnaire avant de le muter.
        """
        rows = [dict(vars(r)) for r in self.results]
        for row in rows:
            row.pop("model", None)
        return pd.DataFrame(rows)

    def get_model(self, period: str, method: str):
        return self.models[f"{period}__{method}"]
