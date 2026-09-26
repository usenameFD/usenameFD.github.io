# Imports et configuration globale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

class IndexReplicator:
    """Construit un portefeuille repliquant un indice par minimisation
    de la variance du Tracking Error.

    Programme :
        min  w' Sigma_TE w
        s.t. sum(w) = 1, 0 <= w_i <= w_max
    """

    def __init__(self, returns, benchmark_returns, long_only=True, w_max=0.10):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.long_only = long_only
        self.w_max = w_max
        self.weights = None
        self.tickers = list(returns.columns)
        self.n = len(self.tickers)

    def fit(self):
        excess = self.returns.sub(self.benchmark_returns, axis=0)
        Sigma_te = excess.cov().values * 252

        def objective(w):
            return w @ Sigma_te @ w

        def gradient(w):
            return 2 * Sigma_te @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bnd = (0.0, self.w_max) if self.long_only else (-self.w_max, self.w_max)
        bounds = [bnd] * self.n

        w0 = np.ones(self.n) / self.n
        result = optimize.minimize(objective, w0, jac=gradient, method='SLSQP',
                                   bounds=bounds, constraints=constraints,
                                   options={'ftol': 1e-10, 'maxiter': 500})

        if not result.success:
            print(f"Attention : optimisation non convergee : {result.message}")

        self.weights = pd.Series(result.x, index=self.tickers, name='weight')
        self.weights = self.weights[self.weights > 1e-5]
        self.weights /= self.weights.sum()

        print(f"Portefeuille : {len(self.weights)} actifs actifs")
        print(f"  Poids max : {self.weights.max():.2%}")
        print(f"  Poids min : {self.weights.min():.2%}")
        print(f"  TE in-sample : {np.sqrt(result.fun):.2%}")
        return self

    def plot_weights(self, top_n=20):
        fig, ax = plt.subplots(figsize=(12, 6))
        top = self.weights.sort_values(ascending=False).head(top_n)
        ax.barh(top.index[::-1], top.values[::-1] * 100, color='orange')
        ax.set_xlabel("Poids (%)")
        ax.set_title(f"Top-{top_n} poids du portefeuille repliquant")
        plt.tight_layout()
        plt.show()

class BacktestEngine:
    """Backtest out-of-sample : Tracking Error, IR, beta, correlation."""

    def __init__(self, weights, returns, benchmark_returns):
        self.weights = weights
        self.returns = returns[weights.index]
        self.benchmark_returns = benchmark_returns
        self.metrics = {}
        self.portfolio_returns = None

    def run(self):
        self.portfolio_returns = self.returns @ self.weights.values
        excess = self.portfolio_returns - self.benchmark_returns

        self.metrics = {
            'TE_annualise': excess.std() * np.sqrt(252),
            'IR': (excess.mean() * 252) / (excess.std() * np.sqrt(252) + 1e-12),
            'Correlation': self.portfolio_returns.corr(self.benchmark_returns),
            'Beta': np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
                    / self.benchmark_returns.var(),
            'Alpha_annualise': (self.portfolio_returns.mean() -
                                self.benchmark_returns.mean()) * 252,
            'Vol_portefeuille': self.portfolio_returns.std() * np.sqrt(252),
            'Vol_benchmark': self.benchmark_returns.std() * np.sqrt(252),
            'Rendement_portefeuille': self.portfolio_returns.mean() * 252,
            'Rendement_benchmark': self.benchmark_returns.mean() * 252,
        }
        return self

    def report(self):
        print("\nRAPPORT DE BACKTEST")
        print("=" * 50)
        for k, v in self.metrics.items():
            if 'Correl' in k or 'Beta' in k:
                print(f"  {k:30s} : {v:>8.4f}")
            else:
                print(f"  {k:30s} : {v:>8.2%}")

    def plot(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        ax = axes[0, 0]
        ((1 + self.portfolio_returns).cumprod() * 100).plot(
            ax=ax, label='Portefeuille', color='orange', lw=2)
        ((1 + self.benchmark_returns).cumprod() * 100).plot(
            ax=ax, label='Benchmark', color='darkblue', ls='--', lw=2)
        ax.set_title("Evolution NAV (base 100)")
        ax.set_ylabel("NAV")
        ax.legend()

        ax = axes[0, 1]
        excess = self.portfolio_returns - self.benchmark_returns
        rolling_te = excess.rolling(60).std() * np.sqrt(252)
        rolling_te.plot(ax=ax, color='orange', lw=2)
        ax.axhline(self.metrics['TE_annualise'], color='darkblue', ls='--',
                   label=f"TE moyen = {self.metrics['TE_annualise']:.2%}")
        ax.set_title("Tracking Error glissante (60j)")
        ax.legend()

        ax = axes[1, 0]
        ax.scatter(self.benchmark_returns, self.portfolio_returns,
                   alpha=0.3, color='orange', s=10)
        lim = max(abs(self.benchmark_returns).max(), abs(self.portfolio_returns).max())
        ax.plot([-lim, lim], [-lim, lim], color='darkblue', ls='--', lw=1)
        ax.set_xlabel("Rendement benchmark")
        ax.set_ylabel("Rendement portefeuille")
        ax.set_title(f"beta = {self.metrics['Beta']:.3f}, "
                     f"rho = {self.metrics['Correlation']:.3f}")

        ax = axes[1, 1]
        excess.hist(bins=80, ax=ax, color='orange', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='darkblue', ls='--')
        ax.set_title("Distribution des ecarts quotidiens")

        plt.tight_layout()
        plt.show()
