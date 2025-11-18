import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class BarrierOption:
    def _init_(self):
        """
        Initialize the BarrierOption object with default values.
        """
        self.strike = None
        self.maturity = None
        self.stock = None
        self.interest_rate = None
        self.sigma = None
        self.option_type = None
        self.barrier = None
        self.price = None
        self.barrier_type = None

    def price_barrier_option(self, initial_price, strike, maturity, interest_rate, volatility, n_simulations, n_steps, barrier, option_type, barrier_type):
        """
        Price barrier options (up-and-out, up-and-in, down-and-out, down-and-in) using Monte Carlo simulations.

        Args:
            initial_price (float): Initial stock price.
            strike (float): Strike price.
            maturity (float): Time to maturity in years.
            interest_rate (float): Risk-free interest rate.
            volatility (float): Stock price volatility.
            n_simulations (int): Number of Monte Carlo simulations.
            n_steps (int): Number of time steps in each simulation.
            barrier (float): Barrier level.
            option_type (str): "call" or "put".
            barrier_type (str): "up-and-out", "up-and-in", "down-and-out", "down-and-in".

        Returns:
            float: The price of the barrier option.
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price

        # Track whether the barrier was hit
        barrier_hit = np.zeros(n_simulations, dtype=bool)

        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = trajectories[:, t - 1] * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)

            if "up" in barrier_type:
                barrier_hit |= trajectories[:, t] >= barrier
            elif "down" in barrier_type:
                barrier_hit |= trajectories[:, t] <= barrier

        # Determine valid trajectories based on barrier type
        if barrier_type == "up-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "up-and-in":
            valid_trajectories = barrier_hit
        elif barrier_type == "down-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "down-and-in":
            valid_trajectories = barrier_hit
        else:
            raise ValueError("Unrecognized barrier type.")

        # Calculate payoffs
        final_prices = trajectories[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError("Unrecognized option type.")

        # Apply barrier condition
        payoffs *= valid_trajectories

        # Discount payoffs back to present value
        option_price = np.exp(-interest_rate * maturity) * np.mean(payoffs)

        # Save attributes for the BarrierOption instance
        self.strike = strike
        self.maturity = maturity
        self.stock = initial_price
        self.interest_rate = interest_rate
        self.sigma = volatility
        self.option_type = option_type
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.price = option_price

        return option_price

    def plot_trajectories(self, initial_price, strike, maturity, interest_rate, volatility, n_simulations, n_steps, barrier, option_type, barrier_type, show_trajectories=50):
        """
        Plot the trajectories of the stock price under the Black-Scholes model with a barrier.

        Args:
            initial_price (float): Initial stock price.
            strike (float): Strike price.
            maturity (float): Time to maturity in years.
            interest_rate (float): Risk-free interest rate.
            volatility (float): Stock price volatility.
            n_simulations (int): Number of Monte Carlo simulations.
            n_steps (int): Number of time steps in each simulation.
            barrier (float): Barrier level.
            option_type (str): "call" or "put".
            barrier_type (str): "up-and-out", "up-and-in", "down-and-out", "down-and-in".
            show_trajectories (int): Number of sample trajectories to display.
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price
        barrier_hit = np.zeros(n_simulations, dtype=bool)

        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = trajectories[:, t - 1] * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)

            if "up" in barrier_type:
                barrier_hit |= trajectories[:, t] >= barrier
            elif "down" in barrier_type:
                barrier_hit |= trajectories[:, t] <= barrier

        # Plot some trajectories
        plt.figure(figsize=(10, 6))
        for i in range(min(show_trajectories, n_simulations)):
            plt.plot(np.linspace(0, maturity, n_steps + 1), trajectories[i], alpha=0.7)
        plt.axhline(y=barrier, color='red', linestyle='--', label='Barrier')
        plt.title("Simulated Trajectories with Barrier")
        plt.xlabel("Time (years)")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_density_and_probabilities(self, initial_price, strike, maturity, interest_rate, volatility, n_simulations, n_steps, barrier, option_type, barrier_type):
        """
        Plot the payoff density and compute the probabilities for barrier options.

        Args:
            initial_price (float): Initial stock price.
            strike (float): Strike price.
            maturity (float): Time to maturity in years.
            interest_rate (float): Risk-free interest rate.
            volatility (float): Stock price volatility.
            n_simulations (int): Number of Monte Carlo simulations.
            n_steps (int): Number of time steps in each simulation.
            barrier (float): Barrier level.
            option_type (str): "call" or "put".
            barrier_type (str): "up-and-out", "up-and-in", "down-and-out", "down-and-in".
        """
        dt = maturity / n_steps
        trajectories = np.zeros((n_simulations, n_steps + 1))
        trajectories[:, 0] = initial_price
        barrier_hit = np.zeros(n_simulations, dtype=bool)

        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_simulations)
            trajectories[:, t] = trajectories[:, t - 1] * np.exp((interest_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * Z)

            if "up" in barrier_type:
                barrier_hit |= trajectories[:, t] >= barrier
            elif "down" in barrier_type:
                barrier_hit |= trajectories[:, t] <= barrier

        if barrier_type == "up-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "up-and-in":
            valid_trajectories = barrier_hit
        elif barrier_type == "down-and-out":
            valid_trajectories = ~barrier_hit
        elif barrier_type == "down-and-in":
            valid_trajectories = barrier_hit

        final_prices = trajectories[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - strike, 0)
        elif option_type == "put":
            payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError("Unrecognized option type.")

        payoffs *= valid_trajectories

        # Compute probabilities
        non_exercise_due_to_barrier = np.sum((payoffs == 0) & barrier_hit)
        if option_type == "call":
            non_exercise_due_to_price = np.sum((payoffs == 0) & (final_prices < strike) & ~barrier_hit)
        elif option_type == "put":
            non_exercise_due_to_price = np.sum((payoffs == 0) & (final_prices > strike) & ~barrier_hit)

        prob_zero_total = np.mean(payoffs == 0)
        prob_non_exercise_barrier = non_exercise_due_to_barrier / n_simulations
        prob_non_exercise_price = non_exercise_due_to_price / n_simulations

        # Display probabilities
        print(f"Probability that the {option_type} option is worthless: {prob_zero_total:.4f}")
        print(f"   - Due to the barrier: {prob_non_exercise_barrier:.4f}")
        print(f"   - Due to the underlying price < strike: {prob_non_exercise_price:.4f}" if option_type == "call" 
              else f"   - Due to the underlying price > strike: {prob_non_exercise_price:.4f}")

        # Plot payoff density
        plt.figure(figsize=(12, 6))
        plt.hist(payoffs, bins=100, density=True, alpha=0.8, color='g', label="Payoff histogram")
        kde = gaussian_kde(payoffs, bw_method='silverman')
        x = np.linspace(min(payoffs), max(payoffs), 600)
        plt.plot(x, kde(x), label="Estimated density", color='red', linewidth=2)
        plt.title(f"Probability Density of the Payoff for a {option_type} Option with Barrier")
        plt.xlabel(f"{option_type.capitalize()} Option Payoff")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
