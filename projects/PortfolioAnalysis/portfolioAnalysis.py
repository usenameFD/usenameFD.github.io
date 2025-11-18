import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioAnalysis:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.025):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.prices = None
        self.returns = None
        self.cov_matrix = None
        self.mu = None  # Expected returns
        self.Sigma = None  # Variance-covariance matrix

    def fetch_data(self):
        """Fetches closing prices for the tickers."""
        self.prices = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        print(self.prices)

    def calculate_returns(self):
        """Calculates total and annualized returns."""
        total_returns = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        self.returns = {
            'total': total_returns * 100,
            'annualized': ((1 + total_returns) ** (1 / 5) - 1) * 100
        }
        print("Rendements totaux sur 5 ans (%):")
        print(self.returns['total'])
        print("\nRendements annualisés (%):")
        print(self.returns['annualized'])

    def calculate_monthly_returns(self):
        """Calculates monthly returns and covariance matrices."""
        monthly_prices = self.prices.resample('M').last()
        monthly_returns = monthly_prices.pct_change()
        self.cov_matrix = monthly_returns.cov() * 12 * 100  # Annualized covariance matrix
        self.mu = monthly_returns.mean() * 12  # Annualized mean returns
        self.Sigma = self.cov_matrix
        print("\nMatrice de variance-covariance annualisée des rendements (en pourcentage) :")
        print(self.cov_matrix)

    def display_correlation_heatmap(self):
        """Displays a heatmap of the correlation matrix."""
        corr_matrix = self.prices.pct_change().corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Monthly Returns')
        plt.show()
        det_corr_matrix = np.linalg.det(corr_matrix)
        if det_corr_matrix != 0:
            print("La matrice de corrélation est inversible.")
        else:
            print("La matrice de corrélation n'est pas inversible.")

    def optimize_portfolio(self, target_return=0.15):
        """Solves for the portfolio weights that minimize risk given a target return."""
        n = len(self.tickers)
        weights = cp.Variable(n)
        risk = cp.quad_form(weights, self.Sigma)
        constraints = [
            cp.sum(weights) == 1,
            weights @ self.mu >= target_return,
            weights >= 0,  # No short selling
            weights <= 1 # Interdiction de vendre à découvert
        ]
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()
        print("Statut de la résolution :", problem.status)
        print("Risque minimal (variance) :", problem.value)
        optimal_weights = pd.Series(weights.value, index=self.tickers)
        print("\nPoids optimaux par actif :")
        print(optimal_weights)
        return weights.value

    def find_tangency_portfolio(self):
        """Finds the tangency portfolio and plots the SML with CAL."""
        mu_targets = np.linspace(0.02, 0.3, 100)
        risks = []
        max_slope = -np.inf
        tangency_weights = None
        tangency_return = None
        tangency_risk = None
        for target in mu_targets:
            weights = cp.Variable(len(self.mu))
            risk = cp.quad_form(weights, self.Sigma)
            constraints = [
                weights @ self.mu >= target,
                cp.sum(weights) == 1
            ]
            problem = cp.Problem(cp.Minimize(risk), constraints)
            problem.solve()
            portfolio_risk = np.sqrt(problem.value)
            risks.append(portfolio_risk)
            slope = (target - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else -np.inf
            if slope > max_slope:
                max_slope = slope
                tangency_weights = weights.value
                tangency_return = target
                tangency_risk = portfolio_risk
        plt.figure(figsize=(10, 6))
        plt.plot(risks, mu_targets, label='SML (Security Market Line)', color='blue', lw=2)
        plt.scatter([tangency_risk], [tangency_return], color='red', label='Portefeuille de Marché', zorder=5)
        plt.plot([0, tangency_risk], [self.risk_free_rate, tangency_return], label='Tangente (CAL)', color='green', linestyle='--')
        plt.xlabel('Risque (Écart-type)', fontsize=12)
        plt.ylabel('Rendement Espéré', fontsize=12)
        plt.title('Security Market Line (SML) et Portefeuille de Marché', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend(fontsize=12)
        plt.show()
        print("Poids du portefeuille de marché :", tangency_weights)
        print("Rendement du portefeuille de marché :", tangency_return)
        print("Risque du portefeuille de marché :", tangency_risk)
