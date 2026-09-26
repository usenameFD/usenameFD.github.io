import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from BondSwap.riskFree import*
from typing import Optional



class Bond:
    def __init__(self) -> None:
        """
        Initialize the Bond class with default parameters.
        """
        self.country: Optional[str] = None
        self.coupon: Optional[float] = None
        self.freq: Optional[float] = None
        self.face_value: Optional[float] = None
        self.riskFree_rate: Optional[callable] = None
        self.price: Optional[float] = None
        self.maturity: Optional[float] = None

    def get_country(self, country: str) -> None:
        """
        Set the country for bond pricing.

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
            riskFree (RiskFree): An instance of the RiskFree class.
        """
        if self.country != riskFree.country:
            raise ValueError(
                f"Mismatch between bond country ({self.country}) and risk-free rate country ({riskFree.country})."
            )
        self.riskFree_rate = riskFree.riskFree_rate

    def get_price(
        self,
        face_value: float,
        coupon_rate: float,
        maturity: float,
        freq: float = 0.5,
    ) -> float:
        """
        Calculate the price of a bond using zero-coupon rates.

        Args:
            face_value (float): The face value (par value) of the bond.
            coupon_rate (float): The annual coupon rate (e.g., 0.05 for 5%).
            maturity (float): The number of years until the bond matures.
            freq (float): Frequency of coupon payments per year (e.g., 1 for annual, 0.5 for semi-annual).

        Returns:
            float: The calculated price of the bond.
        """
        if self.riskFree_rate is None:
            raise ValueError("Risk-free rates are not initialized. Call get_country and get_riskFree_rate first.")

        if face_value <= 0:
            raise ValueError("Face value must be greater than 0.")
        if coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative.")
        if maturity <= 0:
            raise ValueError("Maturity must be greater than 0.")
        if freq <= 0:
            raise ValueError("Frequency must be greater than 0.")

        # Generate target maturities using numpy
        target_maturities = np.arange(freq, maturity + freq, freq)

        # Get zero-coupon rates for the target maturities
        zero_coupon_rates = self.riskFree_rate(target_maturities)

        # Number of payments per year
        m = 1 / freq  # E.g., freq=0.5 means m=2 (semi-annual)
        total_payments = len(target_maturities)

        # Coupon payment per period
        coupon_payment = face_value * coupon_rate / m

        # Validate that there are enough zero-coupon rates
        if len(zero_coupon_rates) < total_payments:
            raise ValueError("Not enough zero-coupon rates provided for all payment periods.")

        # Calculate the present value of coupon payments
        coupon_pv = sum(
            coupon_payment / (1 + zero_coupon_rates[i] / m) ** (i + 1) for i in range(total_payments)
        )

        # Calculate the present value of the face value (final payment)
        face_value_pv = face_value / (1 + zero_coupon_rates[-1] / m) ** total_payments

        # Total price of the bond
        bond_price = coupon_pv + face_value_pv

        # Store bond attributes
        self.coupon = coupon_payment
        self.freq = freq
        self.face_value = face_value
        self.price = bond_price
        self.maturity = maturity

        return bond_price