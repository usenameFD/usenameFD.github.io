import numpy as np
from scipy.stats import norm
from Options.blackScholes import black_scholes_call
from Options.option import Option

class CallOption(Option):
    def __init__(self, 
                 strike: float, 
                 maturity: float, 
                 stock: float, 
                 interest_rate: float, 
                 sigma: float) -> None:
        """
        Initialize a CallOption instance.

        Args:
            strike (float): The strike price of the option.
            maturity (float): Time to maturity in years.
            stock (float): The current stock price.
            interest_rate (float): The risk-free interest rate.
            sigma (float): The volatility of the stock price.
        """
        super().__init__(strike, maturity, stock, interest_rate, sigma)
        self.type: str = "call"  # Explicitly define the type of option

    def option_price(self) -> None:
        """
        Calculate the price of a call option using the Black-Scholes formula.
        """
        self.price = black_scholes_call(self.stock, self.strike, self.interest_rate, self.maturity, self.sigma)

    def delta(self) -> float:
        """
        Calculate the delta of a call option.

        Returns:
            float: The delta of the call option.
        """
        d1 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        return norm.cdf(d1)

    def theta(self) -> float:
        """
        Calculate the theta of a call option.

        Returns:
            float: The theta of the call option.
        """
        d1 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        theta = (-self.stock * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.maturity)) - \
                self.interest_rate * self.strike * np.exp(-self.interest_rate * self.maturity) * norm.cdf(d2)
        return theta

    def rho(self) -> float:
        """
        Calculate the rho of a call option.

        Returns:
            float: The rho of the call option.
        """
        d2 = (np.log(self.stock / self.strike) + 
              (self.interest_rate + 0.5 * self.sigma**2) * self.maturity) / (self.sigma * np.sqrt(self.maturity)) - \
              self.sigma * np.sqrt(self.maturity)
        return self.strike * self.maturity * np.exp(-self.interest_rate * self.maturity) * norm.cdf(d2)