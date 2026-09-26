from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from DRC.code.baseFactorModel import BaseFactorModel
from DRC.code.utils import mean_std_standardizer


class PCAFactorModel(BaseFactorModel):
    """
    Extraction du proxy F du facteur latent Z par analyse en composantes
    principales (chapitre 2, section 2.1.3 du mémoire).

    Version finale : Sigma_Z est fixée à l'identité (chapitre 2, choix
    retenu à l'image de Laurent, Sestier & Thomas 2016). Les scores
    factoriels sont donc STANDARDISÉS (variance unitaire par facteur),
    plutôt que de conserver l'échelle brute Lambda_J des valeurs propres
    de C0 -- cette échelle n'a plus lieu d'être portée par les facteurs,
    puisque Sigma_Z ne varie plus d'une méthode à l'autre.
    """

    def _extract_factors(self) -> None:
        pca = PCA(n_components=self.n_factors)
        scores = pca.fit_transform(self.returns_std.values)
        scores = mean_std_standardizer(scores)  # variance unitaire par facteur

        self.factor_scores = pd.DataFrame(
            scores,
            index=self.returns.index,
            columns=[f"F{k+1}" for k in range(self.n_factors)]
        )

        self.variance_explained = float(np.sum(pca.explained_variance_ratio_))
        self.metadata["explained_variance_ratio"] = pca.explained_variance_ratio_
        self.metadata["components"] = pca.components_