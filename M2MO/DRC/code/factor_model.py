"""Factor Model Calibration (CORRECTION #4, #5)"""
import numpy as np
import warnings

class FactorModelCalibrator:
    """Calibrates factor loadings and idiosyncratic volatility."""
    
    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self.n_assets = None
        self.beta = None
        self.sigma_idio = None
        self.factor_cov = None
        self.explained_variance_ratio = None
    
    def fit(self, returns: 'pd.DataFrame', factor_cov: np.ndarray) -> None:
        """
        Fit factor model via PCA.
        
        CORRECTION #4: Proper idiosyncratic scaling via variance decomposition
        CORRECTION #5: Validate Var(X_i) = 1 constraint
        """
        self.n_assets = returns.shape[1]
        returns_std = (returns - returns.mean()) / returns.std()
        
        U, s, Vt = np.linalg.svd(returns_std.values, full_matrices=False)
        n_factors_actual = min(self.n_factors, returns.shape[1])
        U_factors = U[:, :n_factors_actual]
        s_factors = s[:n_factors_actual]
        
        self.explained_variance_ratio = (s_factors ** 2).sum() / (s ** 2).sum()
        self.beta = (returns.T @ U_factors / len(returns)).values
        self.n_factors = n_factors_actual
        
        # CORRECTION #5: Compute idiosyncratic vol with constraint Var(X_i) = 1
        beta_norm_sq = np.sum(self.beta ** 2, axis=1)
        self.sigma_idio = np.sqrt(np.maximum(1.0 - beta_norm_sq, 0.0))
        self.factor_cov = factor_cov[:n_factors_actual, :n_factors_actual]
        
        self._verify_variance_constraint()
    
    def _verify_variance_constraint(self) -> None:
        """CORRECTION #5: Verify Var(X_i) = 1 for all assets."""
        beta_norm_sq = np.sum(self.beta ** 2, axis=1)
        total_var = beta_norm_sq + self.sigma_idio ** 2
        max_deviation = np.abs(total_var - 1.0).max()
        
        if max_deviation > 0.01:
            warnings.warn(
                f"Variance constraint violated: max deviation = {max_deviation:.4f}",
                UserWarning
            )
        
        print(f"[Factor Model Validation]")
        print(f"  Explained variance: {self.explained_variance_ratio:.2%}")
        print(f"  Var(X_i) - Min: {total_var.min():.6f}, Mean: {total_var.mean():.6f}, Max: {total_var.max():.6f}")
