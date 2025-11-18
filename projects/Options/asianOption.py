import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


class AsianOption:
    def _init_(self):
        """
        Initialize the AsianOption object with default values.
        """
        self.strike: float = None
        self.maturity: float = None
        self.stock: float = None
        self.interest_rate: float = None
        self.sigma: float = None
        self.call_price: float = None
        self.put_price: float = None
        self.call_payoffs: np.ndarray = None
        self.put_payoffs: np.ndarray = None

    def price_asian_option(
        self,
        initial_price: float,
        strike: float,
        maturity: float,
        interest_rate: float,
        volatility: float,
        n_simulations: int,
        n_steps: int,
        averaging_period: float,
        show_trajectories: int = 30,
    ) -> tuple[float, float]:
        """
        Price an Asian option using Monte Carlo simulations.
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price

        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = (
                trajectories[:, t - 1]
                * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)
            )

        n_averaging_steps = int(averaging_period * n_steps)
        averaging_start_index = n_steps + 1 - n_averaging_steps
        averaging_start_time = averaging_start_index * dt

        # Affichage de la ligne rouge (période de moyenne)
        print(f"Averaging start time: {averaging_start_time:.4f} (en années)")  # Affiche la position de la ligne rouge pour débogage

        # Affichage des trajectoires simulées
        plt.figure(figsize=(10, 6))
        for i in range(min(show_trajectories, n_simulations)):
            plt.plot(np.linspace(0, maturity, n_steps + 1), trajectories[i], alpha=0.7)

        plt.axvline(x=averaging_start_time, color='red', linestyle='--', label='Début de la moyenne')
    
        plt.title("Sample Trajectories of Stock Prices")
        plt.xlabel("Time steps")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid()
        plt.show()


        average_prices = np.mean(trajectories[:, averaging_start_index:], axis=1)
        
        self.call_payoffs = np.maximum(average_prices - strike, 0)
        self.put_payoffs = np.maximum(strike - average_prices, 0)

        call_price = np.exp(-interest_rate * maturity) * np.mean(self.call_payoffs)
        put_price = np.exp(-interest_rate * maturity) * np.mean(self.put_payoffs)

        self.strike = strike
        self.maturity = maturity
        self.stock = initial_price
        self.interest_rate = interest_rate
        self.sigma = volatility
        self.call_price = call_price
        self.put_price = put_price


        return call_price, put_price

    def prob_call_zero_and_density(self):
        """Compute probability that call is worthless and plot density."""
        prob_call_zero = np.mean(self.call_payoffs == 0)
        print(f"Probability that Call Payoff is 0: {prob_call_zero:.4f}")
        self._plot_density(self.call_payoffs, "Call Option Payoff Density", "Call Payoff")

    def prob_put_zero_and_density(self):
        """Compute probability that put is worthless and plot density."""
        prob_put_zero = np.mean(self.put_payoffs == 0)
        print(f"Probability that Put Payoff is 0: {prob_put_zero:.4f}")
        self._plot_density(self.put_payoffs, "Put Option Payoff Density", "Put Payoff")

    def _plot_density(self, payoffs: np.ndarray, title: str, xlabel: str):
        """Plot histogram and density estimation of payoffs."""
        plt.figure(figsize=(8, 5))
        hist, bins = np.histogram(payoffs, bins=100, density=True)
        plt.hist(payoffs, bins=100, density=True, alpha=0.6, color='b', label="Histogram")
        kde = gaussian_kde(payoffs, bw_method='silverman')
        x_vals = np.linspace(min(payoffs), max(payoffs), 600)
        plt.plot(x_vals, kde(x_vals), label="Density Estimate", color='r', linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()

