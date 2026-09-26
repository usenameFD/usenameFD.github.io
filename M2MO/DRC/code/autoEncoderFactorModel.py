from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import tensorflow as tf

from DRC.code.baseFactorModel import BaseFactorModel
from DRC.code.utils import  mean_std_standardizer


class AutoEncoderFactorModel(BaseFactorModel):
    def __init__(
        self,
        returns: pd.DataFrame,
        n_factors: int = 2,
        estimate_factor_corr: bool = True,
        hidden_dim: int = 16,
        epochs: int = 350,
        batch_size: int = 32,
        lr: float = 1e-4,
        orthogonalize_factors: bool = False
    ):
        super().__init__(
            returns=returns,
            n_factors=n_factors,
            estimate_factor_corr=estimate_factor_corr
        )

        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.orthogonalize_factors = orthogonalize_factors

    def _orthogonalize(self, Z: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_factors, whiten=True)
        Zw = pca.fit_transform(Z)
        self.metadata["latent_post_pca_explained_variance_ratio"] = pca.explained_variance_ratio_
        return Zw
    
    def _extract_factors(self) -> None:
        Y_train = self.returns.to_numpy()
        scores = extract_autoencoder_factors(Y_train=Y_train, nlatent=self.n_factors, hidden_dim=self.hidden_dim,
                                              val_split=0.2, epochs=self.epochs,
                                                batch_size=self.batch_size, lr=self.lr)
        #scores = mean_std_standardizer(scores)

        # 2. orthogonalisation éventuelle
        if self.orthogonalize_factors:
            scores = self._orthogonalize(scores)
            if self.user_factor_corr is None:
                self.estimate_factor_corr = False

        self.factor_scores = pd.DataFrame(
            scores,
            index=self.returns.index,
            columns=[f"F{k+1}" for k in range(self.n_factors)]
        )

        self.metadata["autoencoder_trained"] = True
        self.metadata["orthogonalize_factors"] = self.orthogonalize_factors


class autoencoder_nonlinear(tf.keras.Model):
    def __init__(self, output_dim, bottleneck_dim=2, hidden_dim=16):
        super(autoencoder_nonlinear, self).__init__()
        self.output_dim = output_dim
        self.bottleneck_dim = bottleneck_dim
        act_fun = "relu"
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation=act_fun),
                tf.keras.layers.Dense(bottleneck_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation=act_fun),
                tf.keras.layers.Dense(output_dim),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def extract_autoencoder_factors(Y_train, nlatent, hidden_dim, val_split, epochs, batch_size, lr):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50)
    model_nonlinear = autoencoder_nonlinear(Y_train.shape[1], nlatent, hidden_dim)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    model_nonlinear.compile(optimizer=adam, loss="mse")
    model_nonlinear.fit(
        x=Y_train,
        y=Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=0,
    )
    factors = model_nonlinear.encoder(Y_train).numpy()
    return factors