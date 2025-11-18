import numpy as np
from scipy.stats import norm
from Options.blackScholes import black_scholes_put
from Options.option import Option

class PutOption(Option):
    def __init__(self, 
                 strike: float, 
                 maturity: float, 
                 stock: float, 
                 interest_rate: float, 
                 sigma: float) -> None:
        """
        Initialize a PutOption instance.

        Args:
            strike (float): The strike price of the option.
            maturity (float): The time to maturity in years.
            stock (float): The current stock price.
            interest_rate (float): The risk-free interest rate.
            sigma (float): The volatility of the stock price.
        """
        super().__init__(strike, maturity, stock, interest_rate, sigma)
        self.type: str = "put"  # Explicitly define the type as a string

    def option_price(self) -> None:
        """
        Calculate the price of a put option using the Black-Scholes formula.
        """
        self.price = black_scholes_put(self.stock, self.strike, self.interest_rate, self.maturity, self.sigma)

    def delta(self) -> float:
        """
        Calculate the delta of a put option.

        Returns:
            float: The delta of the put option.
        """
        d1 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        return norm.cdf(d1) - 1

    def theta(self) -> float:
        """
        Calculate the theta of a put option.

        Returns:
            float: The theta of the put option.
        """
        d1 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        theta = (-self.stock * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.maturity)) + \
                self.interest_rate * self.strike * np.exp(-self.interest_rate * self.maturity) * norm.cdf(-d2)
        return theta

    def rho(self) -> float:
        """
        Calculate the rho of a put option.

        Returns:
            float: The rho of the put option.
        """
        d2 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity)) - \
              self.sigma * np.sqrt(self.maturity)
        return -self.strike * self.maturity * np.exp(-self.interest_rate * self.maturity) * norm.cdf(-d2)