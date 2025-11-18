import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Callable, Optional

def black_scholes_call(stock: float, strike: float, interest_rate: float, maturity: float, sigma: float) -> float:
    """
    Calculate the price of a European call option using the Black-Scholes formula.

    Args:
        stock (float): Current stock price.
        strike (float): Strike price of the option.
        interest_rate (float): Risk-free interest rate (annualized).
        maturity (float): Time to maturity in years.
        sigma (float): Volatility of the underlying stock (annualized).

    Returns:
        float: Price of the European call option.
    """
    # Calculate d1 and d2
    d1 = (np.log(stock / strike) + (interest_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    # Compute the price of the call option
    call_price = stock * norm.cdf(d1) - strike * np.exp(-interest_rate * maturity) * norm.cdf(d2)
    return call_price

def black_scholes_put(stock: float, strike: float, interest_rate: float, maturity: float, sigma: float) -> float:
    """
    Calculate the price of a European put option using the Black-Scholes formula.

    Args:
        stock (float): Current stock price.
        strike (float): Strike price of the option.
        interest_rate (float): Risk-free interest rate (annualized).
        maturity (float): Time to maturity in years.
        sigma (float): Volatility of the underlying stock (annualized).

    Returns:
        float: Price of the European put option.
    """
    # Calculate d1 and d2
    d1 = (np.log(stock / strike) + (interest_rate + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    # Compute the price of the put option
    put_price = strike * np.exp(-interest_rate * maturity) * norm.cdf(-d2) - stock * norm.cdf(-d1)
    return put_price

def implied_volatility(
    option_price: float, 
    stock: float, 
    strike: float, 
    interest_rate: float, 
    maturity: float, 
    black_scholes: Callable[[float, float, float, float, float], float]
) -> Optional[float]:
    """
    Calculate the implied volatility of an option given its price.

    Args:
        option_price (float): Observed market price of the option.
        stock (float): Current stock price.
        strike (float): Strike price of the option.
        interest_rate (float): Risk-free interest rate (annualized).
        maturity (float): Time to maturity in years.
        black_scholes (Callable): Pricing function for the option (call or put).

    Returns:
        Optional[float]: The implied volatility if it converges; otherwise, None.
    """
    # Function to minimize: Difference between market price and Black-Scholes price
    func = lambda sigma: black_scholes(stock, strike, interest_rate, maturity, sigma) - option_price

    # Attempt to find the implied volatility using Brent's method
    try:
        return brentq(func, 1e-16, 5)  # Volatility range from near-zero to 500%
    except ValueError:
        # Return None if no solution is found
        return None