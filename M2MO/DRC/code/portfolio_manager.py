"""
Portfolio Data Manager - Handles portfolio state and market values for DRC calculation

CORRECTION #1 & #2:
- JTD must be CURRENT MARKET VALUES, not optimization weights
- LGD must be equity-specific (60-100%), not bond table (40-60%)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings


class PortfolioDataManager:
    """
    Manages portfolio state and market values for DRC calculation.
    
    KEY CORRECTIONS:
    ================
    1. JTD = current market value (shares × current_price), updated dynamically
    2. LGD from equity-specific table, not bond LGD table
    
    Mathematical Foundation:
    =========================
    JTD_i(t_calc) = Shares_i × Price_i(t_calc)  [CORRECT]
    NOT: JTD_i ≠ w_i(t_0) × Notional(t_0)      [INCORRECT - stale]
    
    This ensures:
    - Exposure reflects CURRENT market value
    - P&L changes are captured
    - Volatility in prices correctly changes JTD
    """
    
    def __init__(self,
                 initial_positions: Dict[str, float],
                 initial_prices: Dict[str, float],
                 ratings: Dict[str, str],
                 pd_table: Dict[str, float],
                 lgd_table_equity: Dict[str, float]):
        """
        Initialize portfolio manager.
        
        Parameters
        ----------
        initial_positions : dict
            Number of shares held per asset {ticker: shares}
        initial_prices : dict
            Prices at portfolio initialization {ticker: price}
        ratings : dict
            Credit ratings by ticker {ticker: rating}
        pd_table : dict
            PD lookup by rating {rating: PD}
        lgd_table_equity : dict
            LGD for equities (CORRECTED: not bond table)
            Typical: {'AAA': 0.6, 'AA': 0.65, 'A': 0.7, 'BBB': 0.8, 'BB': 1.0}
        """
        self.shares = initial_positions.copy()
        self.ratings = ratings.copy()
        self.pd_table = pd_table.copy()
        
        # CORRECTED: Use equity-specific LGD table (not bond LGD)
        self.lgd_table_equity = lgd_table_equity.copy()
        
        self.current_prices = initial_prices.copy()
        self.initial_prices = initial_prices.copy()
        self.tickers = list(initial_positions.keys())
        self.n_assets = len(self.tickers)
        
        # Validation
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input data consistency."""
        # Check all tickers are present in all tables
        for ticker in self.tickers:
            if ticker not in self.ratings:
                raise ValueError(f"Ticker {ticker} missing from ratings")
            if self.ratings[ticker] not in self.pd_table:
                raise ValueError(f"Rating {self.ratings[ticker]} not in PD table")
            if self.ratings[ticker] not in self.lgd_table_equity:
                raise ValueError(f"Rating {self.ratings[ticker]} not in LGD equity table")
        
        # Check LGD values are in [0,1]
        for rating, lgd in self.lgd_table_equity.items():
            if not (0 <= lgd <= 1):
                raise ValueError(f"LGD for {rating}: {lgd} not in [0,1]")
        
        # Check LGD for equities ≈ 0.6-1.0 (not 0.4-0.6 like bonds)
        avg_lgd = np.mean(list(self.lgd_table_equity.values()))
        if avg_lgd < 0.5:
            warnings.warn(
                f"Average LGD is {avg_lgd:.2%}, which is low for equities. "
                f"Did you use a bond LGD table by mistake? "
                f"Equity LGD should be 0.6-1.0.",
                UserWarning
            )
    
    def update_prices(self, new_prices: Dict[str, float]) -> None:
        """
        Update market prices for all assets.
        
        CRITICAL: Call this before each DRC calculation to get current JTD
        
        Parameters
        ----------
        new_prices : dict
            Current prices by ticker {ticker: price}
        """
        for ticker in self.tickers:
            if ticker in new_prices:
                self.current_prices[ticker] = new_prices[ticker]
    
    def get_jtd_array(self) -> np.ndarray:
        """
        Compute Jump-To-Default exposure for each asset.
        
        MATHEMATICAL CORRECTION:
        ========================
        JTD_i(t_calc) = Shares_i × Price_i(t_calc)  [CORRECT]
        NOT: JTD_i = w_i(t_0) × Notional           [INCORRECT]
        
        Returns
        -------
        np.ndarray
            JTD values for each asset (shape: (n_assets,))
        """
        # CORRECTED: Use current market values
        jtd = np.array([
            self.shares[ticker] * self.current_prices[ticker]
            for ticker in self.tickers
        ])
        return jtd
    
    def get_pd_array(self) -> np.ndarray:
        """Get probability of default for each asset."""
        pd_array = np.array([
            self.pd_table[self.ratings[ticker]]
            for ticker in self.tickers
        ])
        return pd_array
    
    def get_lgd_array(self) -> np.ndarray:
        """
        Get Loss-Given-Default for each asset.
        
        MATHEMATICAL CORRECTION:
        ========================
        INCORRECT: Use bond LGD table (lgd_table_sp) → 40-60%
        CORRECT: Use equity-specific LGD → 60-100%
        
        Rationale:
        - Bond default → recovery from restructuring/collateral
        - Equity default → stock → €0 (shareholders get nothing)
        - Regulatory (CRR3): LGD for equity = 60-100%
        
        Returns
        -------
        np.ndarray
            LGD values for each asset (shape: (n_assets,))
        """
        # CORRECTED: Use equity LGD table, not bond table
        lgd_array = np.array([
            self.lgd_table_equity[self.ratings[ticker]]
            for ticker in self.tickers
        ])
        return lgd_array
    
    def get_total_notional(self) -> float:
        """Return total portfolio notional (current market value)."""
        return self.get_jtd_array().sum()
    
    def report_portfolio_state(self, label: str = "Portfolio State") -> None:
        """Print current portfolio composition and exposures."""
        print(f"\n{'='*70}")
        print(f"{label}")
        print(f"{'='*70}")
        print(f"{'Ticker':<8} {'Shares':>12} {'Price':>12} {'JTD':>15} {'Rating':>8}")
        print("-" * 70)
        
        jtd_array = self.get_jtd_array()
        for i, ticker in enumerate(self.tickers):
            print(f"{ticker:<8} {self.shares[ticker]:>12.0f} "
                  f"{self.current_prices[ticker]:>12.2f} "
                  f"{jtd_array[i]:>15.2f} {self.ratings[ticker]:>8}")
        
        print("-" * 70)
        print(f"{'TOTAL NOTIONAL':.<50} {self.get_total_notional():>15.2f}")
        print(f"{'='*70}")
