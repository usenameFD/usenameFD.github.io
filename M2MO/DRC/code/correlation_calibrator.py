"""Correlation Calibration with EWMA (CORRECTION #3)"""
import numpy as np
import warnings

class CorrelationCalibrator:
    """Calibrates factor correlations using exponentially-weighted moving average."""
    
    def __init__(self, lookback_days: int = 120, decay_factor: float = 0.94):
        """
        CORRECTION #3: EWMA instead of 5-year rolling window
        
        Parameters
        ----------
        lookback_days : int
            Number of trading days for calibration window (default: 120 ≈ 6 months)
        decay_factor : float
            Exponential decay (default: 0.94 = RiskMetrics standard)
            λ=0.94 means 1-week-old data gets 94% weight relative to today
        """
        self.lookback_days = lookback_days
        self.decay_factor = decay_factor
        
        if not (0.85 < decay_factor < 1.0):
            raise ValueError(f"decay_factor must be in (0.85, 1.0), got {decay_factor}")
    
    def compute_ewma_covariance(self, returns: 'pd.DataFrame') -> np.ndarray:
        """
        Compute exponentially-weighted moving average covariance matrix.
        
        Formula:
        Σ(t) = [Σ_s λ^{T-s} r_s r_s^T] / [Σ_s λ^{T-s}]
        
        CORRECTION: Recent data gets higher weight, captures volatility clustering
        """
        if len(returns) > self.lookback_days:
            returns = returns.iloc[-self.lookback_days:].copy()
        
        returns_centered = returns - returns.mean()
        T = len(returns_centered)
        
        # Exponential weights: λ^{T-1-s} (higher weight for recent s)
        weights = np.array([self.decay_factor ** (T - 1 - s) for s in range(T)])
        weights = weights / weights.sum()
        
        # Weighted covariance
        cov_matrix = np.zeros((returns_centered.shape[1], returns_centered.shape[1]))
        for s in range(T):
            r_s = returns_centered.iloc[s].values
            cov_matrix += weights[s] * np.outer(r_s, r_s)
        
        return cov_matrix
    
    def ensure_psd(self, matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive-definite by eigenvalue clipping."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        matrix_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return matrix_psd
