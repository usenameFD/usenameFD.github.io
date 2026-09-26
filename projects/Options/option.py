import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod


class Option(ABC):
    """
    Abstract base class for financial options. Provides methods for calculating common greeks 
    and requires subclasses to implement pricing and specific greeks (delta, theta, rho).
    """

    def __init__(self, strike: float, maturity: float, stock: float, interest_rate: float, sigma: float):
        """
        Initialize the Option object.

        Args:
            strike (float): The strike price of the option.
            maturity (float): Time to maturity in years.
            stock (float): Current stock price.
            interest_rate (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying stock.

        Raises:
            ValueError: If any input is negative or invalid.
        """
        if strike <= 0:
            raise ValueError("Strike price must be positive.")
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if stock <= 0:
            raise ValueError("Stock price must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive.")

        self.strike = strike
        self.maturity = maturity
        self.stock = stock
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.price = None  # Will be set in subclass implementations
        self.type = None  # "call" or "put", set by subclasses

    def _d1(self) -> float:
        """
        Calculate the d1 term used in Black-Scholes formulas.

        Returns:
            float: The d1 value.
        """
        return (np.log(self.stock / self.strike) +
                (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))

    def _d2(self) -> float:
        """
        Calculate the d2 term used in Black-Scholes formulas.

        Returns:
            float: The d2 value.
        """
        return self._d1() - self.sigma * np.sqrt(self.maturity)

    def gamma(self) -> float:
        """
        Calculate the gamma of the option.

        Returns:
            float: Gamma of the option.
        """
        d1 = self._d1()
        return norm.pdf(d1) / (self.stock * self.sigma * np.sqrt(self.maturity))

    def vega(self) -> float:
        """
        Calculate the vega of the option.

        Returns:
            float: Vega of the option.
        """
        d1 = self._d1()
        return self.stock * np.sqrt(self.maturity) * norm.pdf(d1)

    @abstractmethod
    def option_price(self) -> float:
        """
        Abstract method to calculate the price of the option.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def delta(self) -> float:
        """
        Abstract method to calculate the delta of the option.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def theta(self) -> float:
        """
        Abstract method to calculate the theta of the option.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def rho(self) -> float:
        """
        Abstract method to calculate the rho of the option.
        Must be implemented by subclasses.
        """
        pass