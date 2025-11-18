import pandas as pd
import numpy as np
from typing import List, Union

from BondSwap.riskFree import*

class Swap:
    def __init__(self) -> None:
        """
        Initialize the Swap class with default parameters.
        """
        self.country: Union[str, None] = None  # Country code ('usa' or 'fr')
        self.riskFree_rate: Union[callable, None] = None  # Risk-free rate function
        self.swap_rate: Union[float, None] = None  # Implied swap rate
        self.swap_value: Union[float, None] = None  # Net present value of the swap

    def get_country(self, country: str) -> None:
        """
        Set the country for the swap.

        Args:
            country (str): The country ("usa" or "fr").
        """
        if country.lower() not in ["usa", "fr"]:
            raise ValueError("Unsupported country. Use 'usa' or 'fr'.")
        self.country = country.lower()

    def get_riskFree_rate(self, riskFree: object) -> None:
        """
        Set the risk-free rate function based on the given RiskFree object.

        Args:
            riskFree (object): An instance of the RiskFree class.
        """
        if riskFree.country != self.country:
            raise ValueError(
                f"Mismatch between swap country ({self.country}) and risk-free rate country ({riskFree.country})."
            )
        self.riskFree_rate = riskFree.riskFree_rate

    def B(self, t: float, T: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        Calculate the discount factor from time t to T using the yield curve.

        Args:
            t (float): Current time.
            T (float or array-like): Future time(s) for which to calculate the discount factor.

        Returns:
            np.ndarray: Discount factor(s) for the given time(s).
        """
        if self.riskFree_rate is None:
            raise ValueError("Risk-free rate is not initialized. Call get_riskFree_rate first.")

        T = np.array(T, dtype=float)  # Ensure T is an array
        if np.any(T < t):
            raise ValueError("All future times (T) must be greater than or equal to the current time (t).")

        # Get zero-coupon rates
        zero_coupon_rates = self.riskFree_rate(T)

        # Calculate discount factors
        return np.exp(-zero_coupon_rates * (T - t))

    def FRA(self, t: float, T1: float, T2: float) -> float:
        """
        Calculate the forward rate agreement (FRA) between times T1 and T2.

        Args:
            t (float): Current time.
            T1 (float): Start of the forward period.
            T2 (float): End of the forward period.

        Returns:
            float: Forward rate between T1 and T2.
        """
        if T1 < t or T2 <= T1:
            raise ValueError("Ensure that t <= T1 < T2.")

        discount_T1 = self.B(t, T1) if t != T1 else 1.0
        discount_T2 = self.B(t, T2)
        forward_rate = (1 / (T2 - T1)) * (discount_T1 / discount_T2 - 1)
        return forward_rate

    def swap(self, t: float, T: List[float], N: float, K: float) -> float:
        """
        Calculate the present value of a swap.

        Args:
            t (float): Current time.
            T (list of float): Array of payment times (e.g., T = [1, 2, 3, ..., n]).
            N (float): Notional amount of the swap.
            K (float): Fixed swap rate.

        Returns:
            float: Net present value of the swap.
        """
        if self.riskFree_rate is None:
            raise ValueError("Risk-free rate is not initialized. Call get_riskFree_rate first.")

        T = np.array(T, dtype=float)  # Ensure T is a numpy array
        if len(T) < 2:
            raise ValueError("At least two payment times are required.")
        if not np.all(np.diff(T) > 0):
            raise ValueError("Payment times must be in ascending order.")

        # Add t to the start of T
        T = np.insert(T, 0, t)

        # Calculate discount factors for all payment times
        discount_factors = self.B(t, T[1:])  # Exclude the first element (t itself)

        # Calculate forward rates for each period
        forward_rates = np.array([self.FRA(t, T[i - 1], T[i]) for i in range(1, len(T))])

        # Calculate time deltas between payments
        deltas = np.diff(T)

        # Calculate the floating and fixed legs
        floating_leg = np.sum(deltas * discount_factors * forward_rates)
        fixed_leg = np.sum(deltas * discount_factors * K)

        # Net present value of the swap
        swap_value = N * (floating_leg - fixed_leg)

        # Implied swap rate
        swap_rate = (discount_factors[0] - discount_factors[-1]) / np.sum(deltas * discount_factors)

        self.swap_value = swap_value
        self.swap_rate = swap_rate
        return swap_value
