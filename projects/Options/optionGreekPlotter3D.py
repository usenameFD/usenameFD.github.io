import numpy as np
import plotly.graph_objects as go
from typing import Callable, Any

from Options.callOption import CallOption
from Options.putOption import PutOption

class OptionGreekPlotter3D:
    def __init__(self, stock: float, sigma: float, interest_rate: float, type_option: str):
        """
        Initialize the OptionGreekPlotter3D with common parameters for the options.

        Parameters:
        - stock (float): Current stock price.
        - sigma (float): Volatility of the options.
        - interest_rate (float): Risk-free interest rate.
        - type_option (str): The type of option, either 'call' or 'put'.
        """
        self.stock = stock
        self.sigma = sigma
        self.interest_rate = interest_rate
        self.type = type_option

    def compute_greeks_3d(self, greek_function: Callable[[Any], float], strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        """
        Compute Greek values for call and put options over a grid of strikes and maturities.

        Parameters:
        - greek_function (Callable): A function to compute the Greek for an option.
        - strikes (np.ndarray): Array of strike prices.
        - maturities (np.ndarray): Array of maturities.

        Returns:
        - greek (np.ndarray): 2D array of Greek values for options.
        """
        greek = np.zeros((len(strikes), len(maturities)))

        for i, strike in enumerate(strikes):
            for j, T in enumerate(maturities):
                if self.type == "call":
                    option = CallOption(strike, T, self.stock, self.interest_rate, self.sigma)
                elif self.type == "put":
                    option = PutOption(strike, T, self.stock, self.interest_rate, self.sigma)
                greek[i, j] = greek_function(option)
        return greek

    def plot_greeks(self, greek_function: Callable[[Any], float], strikes: np.ndarray, maturities: np.ndarray, greek_name: str):
        """
        Create interactive 3D and dynamic 2D plots for Greeks.

        Parameters:
        - greek_function (Callable): A function to compute the Greek for an option.
        - strikes (np.ndarray): Array of strike prices.
        - maturities (np.ndarray): Array of maturities.
        - greek_name (str): Name of the Greek being plotted.
        """
        greek = self.compute_greeks_3d(greek_function, strikes, maturities)

        # 3D Plot
        fig = go.Figure()

        # Surface for Options
        fig.add_trace(go.Surface(
            z=greek.T,
            x=strikes,
            y=maturities,
            colorscale='Viridis',
            name=f'{self.type} Options'
        ))

        # Show the 3D plot
        fig.update_layout(
            title=f"{greek_name} Evolution",
            scene=dict(
                xaxis_title='Strike Price',
                yaxis_title='Maturity (T)',
                zaxis_title=f"{greek_name}",
                aspectratio=dict(x=1.5, y=1.5, z=0.5)
            )
        )

        fig.show()