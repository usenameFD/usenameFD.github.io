import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Callable, Tuple

from Options.callOption import CallOption
from Options.putOption import PutOption

class OptionGreekPlotter:
    def __init__(self, strike: float, T: float, stock: float, sigma: float, interest_rate: float):
        """
        Initialize the OptionGreekPlotter with common parameters for the options.

        Parameters:
        - strike (float): Strike price of the options.
        - T (float): Maturity of the options.
        - stock (float): Current stock price.
        - sigma (float): Volatility of the options.
        - interest_rate (float): Risk-free interest rate.
        """
        self.strike = strike
        self.T = T
        self.stock = stock
        self.sigma = sigma
        self.interest_rate = interest_rate

    def compute_greek(self, greek_function: Callable, x_values: np.ndarray, x_label: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Greek value for call and put options over a range of X values.

        Parameters:
        - greek_function (Callable): A function to compute the Greek for an option.
        - x_values (np.ndarray): An array of X values (e.g., stock prices, maturities, etc.).
        - x_label (str): The X-axis variable being varied.

        Returns:
        - call_values (np.ndarray): Computed Greek values for the call options.
        - put_values (np.ndarray): Computed Greek values for the put options.
        """
        call_values = np.zeros_like(x_values)
        put_values = np.zeros_like(x_values)

        for i, x in enumerate(x_values):
            # Update the parameter based on the X-axis label
            strike, T, sigma, interest_rate = self.strike, self.T, self.sigma, self.interest_rate
            if x_label == "Strike":
                strike = x
            elif x_label == "Maturity (T)":
                T = x
            elif x_label == "Sigma (Volatility)":
                sigma = x
            elif x_label == "Interest Rate":
                interest_rate = x

            # Create call and put options
            call_option = CallOption(strike, T, self.stock, interest_rate, sigma)
            put_option = PutOption(strike, T, self.stock, interest_rate, sigma)

            # Compute the Greek
            call_values[i] = greek_function(call_option)
            put_values[i] = greek_function(put_option)

        return call_values, put_values

    def plot_greek(self, greek_function: Callable, x_values: np.ndarray, x_label: str, y_label: str, title: str):
        """
        Plot a Greek for both call and put options over a range of X values.

        Parameters:
        - greek_function (Callable): A function to compute the Greek for an option.
        - x_values (np.ndarray): An array of X values (e.g., stock prices, maturities, etc.).
        - x_label (str): Label for the X-axis.
        - y_label (str): Label for the Y-axis.
        - title (str): Title for the plot.
        """
        call_values, put_values = self.compute_greek(greek_function, x_values, x_label)

        # Create the plot
        fig = go.Figure()

        # Add Call line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=call_values,
            mode='lines',
            name=f"Call {y_label}",
            line=dict(color='blue')
        ))

        # Add Put line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=put_values,
            mode='lines',
            name=f"Put {y_label}",
            line=dict(color='red')
        ))

        # Customize layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            legend=dict(title='Options'),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        # Show the interactive plot
        fig.show()