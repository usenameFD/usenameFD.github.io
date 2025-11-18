import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
from Options.blackScholes import implied_volatility, black_scholes_call, black_scholes_put


class OptionsData:
    """
    A class to collect and process options data, estimate implied volatility,
    and visualize results.
    """

    def __init__(self, ticker: str, extract_date: pd.Timestamp = pd.Timestamp.today()) -> None:
        """
        Initialize the OptionsData class.

        Args:
            ticker (str): Stock ticker symbol.
            extract_date (pd.Timestamp, optional): Date for data extraction. Defaults to today.
        """
        self.ticker = ticker
        self.extract_date = extract_date
        self.data = None  # This will store the combined options DataFrame
        self.implied_volatility = None

    def load_data(self) -> None:
        """
        Load options data (calls and puts) for all available expiration dates.
        """
        ticker = yf.Ticker(self.ticker)

        try:
            # Fetch available expiration dates
            options_dates = ticker.options
            if not options_dates:
                raise ValueError("No options data available for this ticker.")

            df_list = []

            for maturity in options_dates:
                # Fetch the option chain for the given maturity
                chain = ticker.option_chain(maturity)
                calls, puts = chain.calls, chain.puts

                # Add "type" and "maturity" columns
                calls["type"], puts["type"] = "call", "put"
                calls["maturity"], puts["maturity"] = maturity, maturity

                # Combine calls and puts
                df_list.append(calls)
                df_list.append(puts)

            # Combine all maturities into a single DataFrame
            self.data = pd.concat(df_list, ignore_index=True)
            print("Data loaded successfully!")

        except Exception as e:
            print(f"Error while loading data: {e}")

    def process_data(self, interest_rate: float = 0.042) -> None:
        """
        Process the loaded options data and calculate implied volatility.

        Args:
            interest_rate (float): Risk-free interest rate. Defaults to 0.042 (4.2%).
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()
        df["maturity"] = pd.to_datetime(df["maturity"])
        df["maturity"] = (df["maturity"] - self.extract_date).dt.days / 365

        # Group by strike, maturity, and type, and aggregate
        df = df.groupby(["strike", "maturity", "type"], as_index=False).agg(
            {"lastPrice": "first", "volume": "sum"}
        )

        # Select rows with the maximum volume per strike/maturity pair
        df = df.loc[df.groupby(["strike", "maturity"])["volume"].idxmax()]

        # Filter options with sufficient trading volume
        df = df[df["volume"] >= 10]

        # Fetch the stock's current price
        stock_today = yf.Ticker(self.ticker).history(period="1d")["Close"].iloc[-1]

        # Separate calls and puts
        call_data = df[df["type"] == "call"]
        put_data = df[df["type"] == "put"]

        # Calculate implied volatility for calls and puts
        call_data["implied_volatility"] = call_data.apply(
            lambda row: implied_volatility(
                option_price=row["lastPrice"],
                stock=stock_today,
                strike=row["strike"],
                interest_rate=interest_rate,
                maturity=row["maturity"],
                black_scholes=black_scholes_call,
            ),
            axis=1,
        )

        put_data["implied_volatility"] = put_data.apply(
            lambda row: implied_volatility(
                option_price=row["lastPrice"],
                stock=stock_today,
                strike=row["strike"],
                interest_rate=interest_rate,
                maturity=row["maturity"],
                black_scholes=black_scholes_put,
            ),
            axis=1,
        )

        self.implied_volatility = pd.concat([call_data, put_data], ignore_index=True)

    def implied_volatility_plot(self) -> None:
        """
        Plot the implied volatility surface as a 3D graph.
        """
        if self.implied_volatility is None:
            raise ValueError("Implied volatility data not processed. Call process_data() first.")

        df = self.implied_volatility.dropna(subset=["implied_volatility"])
        df = df[df["maturity"] > (20 / 365)]  # Exclude maturities < 30 days

        # Create a grid for interpolation
        T_grid, K_grid = np.meshgrid(
            np.linspace(df["maturity"].min(), df["maturity"].max(), 100),
            np.linspace(df["strike"].min(), df["strike"].max(), 100),
        )

        vol_grid = griddata(
            (df["maturity"], df["strike"]),
            df["implied_volatility"],
            (T_grid, K_grid),
            method="linear",
        )

        # Plot the implied volatility surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            K_grid, T_grid, vol_grid, cmap="viridis", edgecolor="none", alpha=0.8
        )

        ax.set_xlabel("Strike Price (K)")
        ax.set_ylabel("Time to Maturity (T, in years)")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Implied Volatility Surface")

        cbar = fig.colorbar(surf, ax=ax, pad=0.1)
        cbar.set_label("Implied Volatility")

        plt.show()

    def sigma_sim(self, T: float, K: float) -> float:
        """
        Get the implied volatility for a given maturity and strike.

        Args:
            T (float): Time to maturity in years.
            K (float): Strike price.

        Returns:
            float: Implied volatility for the given maturity and strike.
        """
        if self.implied_volatility is None:
            raise ValueError("Implied volatility data not processed. Call process_data() first.")

        df = self.implied_volatility.dropna(subset=["implied_volatility"])
        df = df[df["maturity"] > (20 / 365)]  # Exclude short maturities

        sigma = griddata(
            (df["maturity"], df["strike"]),
            df["implied_volatility"],
            (T, K),
            method="linear",
        )

        return sigma

    def get_stock_price(self) -> float:
        """
        Get the current stock price.

        Returns:
            float: Current stock price.
        """
        stock_today = yf.Ticker(self.ticker).history(period="1d")["Close"].iloc[-1]
        return stock_today
