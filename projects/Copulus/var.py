import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
from skew_student import optimize_parameters, skew_student_sim
from scipy.optimize import minimize
from scipy.stats import genextreme, gumbel_r, genpareto
from arch import arch_model


import scipy.stats as stats
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, chi2
    

class Var:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def load_data(self):
        """Load historical data for the given ticker and calculate log returns."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)["Close"]
        data = pd.DataFrame(data)
        data.columns = ["Close"]
        data.index = pd.to_datetime(data.index)
        data["return"] = np.log(data["Close"] / data["Close"].shift(1))
        data.dropna(inplace=True)
        self.data = data
        
    def plot_data(self):
        """Plot historical returns and closing prices using Plotly."""
        fig = go.Figure()

        # First y-axis for returns
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data["return"],
            mode='lines',
            name=f'Hist Return {self.ticker}',
            line=dict(color='red'),
        ))

        # Second y-axis for closing prices
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data["Close"],
            mode='lines',
            name=f'Hist Close {self.ticker}',
            line=dict(color='blue'),
            yaxis='y2',  # This ensures that the second trace uses the second y-axis
        ))

        # Update layout to add a second y-axis
        fig.update_layout(
            title=f'{self.ticker} Historical Data',
            xaxis_title='Date',
            yaxis=dict(
                title='Return',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
            ),
            yaxis2=dict(
                title='Close',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                overlaying='y',  # This ensures the second y-axis shares the same x-axis
                side='right',    # The second y-axis will be on the right side
            ),
            showlegend=True,
            template='plotly_dark',  # Optional, change template as needed
        )

        return fig
    
    def train_test_split(self, start_train, start_test, end_test):
        """Split data into training and testing sets."""
        end_train =  pd.Timestamp(start_test) - pd.Timedelta(days=1)
        data_train = self.data.loc[start_train:end_train]
        data_test = self.data.loc[start_test:end_test]
        return data_train, data_test
    
    def Var_Hist(self, data, alpha):
        """Calculate Historical Value at Risk (VaR) and Expected Shortfall (ES)."""
        VaR = data.quantile(1 - alpha).iloc[0]
        ES = data.loc[data["return"] < VaR, "return"].mean()
        return {"VaR": VaR, "ES": float(ES)}

    def Var_Hist_Bootstrap(self, data, alpha, B, alpha_IC, M):
        """Calculate VaR using bootstrap method with confidence intervals."""
        var = []
        for _ in range(M):
            index = np.random.choice(data.index, size=B, replace=True)  # Bootstrap sampling with replacement
            var.append(data.loc[index, "return"].quantile(1 - alpha))
        
        var = np.array(var)
        alpha_IC_bis = (1 - alpha_IC) / 2
        b_inf = np.percentile(var, alpha_IC_bis * 100)  # Lower bound of confidence interval
        b_sup = np.percentile(var, (1 - alpha_IC_bis) * 100)  # Upper bound of confidence interval

        return {'VaR': var.mean(),
                f'IC_lower_{round(1 - alpha_IC, 2)}': b_inf,
                f'IC_upper_{alpha_IC}': b_sup}
    
    def Var_param_gaussian(self, data, alpha):
        """Calculate VaR using Gaussian distribution."""
        mu = data.mean()
        sigma = data.std()
        Z = mu + sigma * np.random.normal(0, 1, len(data))
        Z = pd.DataFrame(Z, index=data.index, columns=["return"])
        return Z
    
    # On calcule la VaR à horizon 10 jours par diffusion du cours du CAC40
    def simulate_price_paths(self, t, S0, mu, sigma, num_simulations):
        # Créer une matrice vide pour stocker les prix simulés à chaque étape
        St = np.zeros((num_simulations, t))

        # Simuler les trajectoires de prix
        for i in range(num_simulations):
            # Initialiser le premier prix à S0
            St[i, 0] = S0

            # Générer des variables aléatoires Z de loi normale standard
            Z = np.random.normal(0, 1, t)
            #print(Z)

            # Calculer les prix simulés à chaque étape
            for j in range(1, t):

                St[i, j] = St[i, j-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.sqrt(1) * Z[j-1])

        return St

    ## ii - On calcule les log rendements à horizon 10 jours pour chacune des trajectoires

    def calculate_log_returns(self, St, S0):
        # Calculer les log returns
        S0_scalar = S0
        log_returns = np.log(St[:, -1] / S0_scalar)  # Calcul des rendements log au bout de t jours
        return log_returns

        ### iii- On en déduit la valeur de la VaR 
    
    def calculate_var(self, log_returns, confidence_level=0.99):
        # Calculer le quantile d'ordre (1 - percentile) de la distribution des pertes
        var = np.percentile(log_returns, 100 * (1 - confidence_level))
        return var
    
    def calculate_var_diffusion(self, data_train, horizon = 10, alpha=0.99, num_simulations = 10_00):
        S0 = data_train['Close'].iloc[-1]
        mu = np.mean(data_train['return'])
        sigma = np.std(data_train['return'])
        St = self.simulate_price_paths(horizon+1, S0, mu, sigma, num_simulations)
        
        # Calcul des rendements log
        log_returns = self.calculate_log_returns(St, S0)
        log_returns = pd.DataFrame(log_returns, columns=["return"])
        var = self.Var_Hist(log_returns, alpha)
        
        return var


    ## Protocole de backtesting
    
    def perform_backtest(self, data_test, var_99, significance_level_var=0.01, significance_test=0.05):
        """
        Effectue les tests d'unconditional coverage et d'indépendance sur un modèle de VaR.
        
        - Unconditional Coverage : Vérifie si la proportion d'excès observés correspond au seuil théorique (par exemple, 1% pour une VaR à 99%).
        - Indépendance : Vérifie si les excès observés sont indépendants entre eux.
        
        Retourne True si les tests sont validés (p-value > seuil de signification), sinon False.
        
        Parameters:
        -----------
        - data_test : pandas.DataFrame
            DataFrame contenant les rendements log des actifs.
        - var_99 : float
            Valeur de la VaR à 99%.
        - significance_level_var : float, optionnel (default=0.01)
            Niveau de signification pour le test d'Unconditional Coverage.
        - significance_test : float, optionnel (default=0.05)
            Niveau de signification pour les tests statistiques.

        Returns:
        --------
        - valid_tests : bool
            True si les deux tests sont validés, False sinon.
        - n_exceed : int
            Nombre d'excès observés.

        """
        # Test d'Unconditional Coverage
        data_test['VaR_exceedance'] = data_test['return'] < var_99
        n_exceed = data_test['VaR_exceedance'].sum()
        obs = len(data_test)
        prob_emp = n_exceed / obs
        LR_uc = -2 * np.log(((1 - significance_level_var) ** (obs - n_exceed)) * (significance_level_var ** n_exceed)) + \
                2 * np.log(((1 - prob_emp) ** (obs - n_exceed)) * (prob_emp ** n_exceed))
        p_value_uc = 1 - chi2.cdf(LR_uc, df=1)

        # Test d'Indépendance
        data_test['T_0_1'] = ((data_test['VaR_exceedance'].shift(1) == 0) & (data_test['VaR_exceedance'] == 1)).astype(int)
        data_test['T_1_0'] = ((data_test['VaR_exceedance'].shift(1) == 1) & (data_test['VaR_exceedance'] == 0)).astype(int)
        data_test['T_1_1'] = ((data_test['VaR_exceedance'].shift(1) == 1) & (data_test['VaR_exceedance'] == 1)).astype(int)
        data_test['T_0_0'] = ((data_test['VaR_exceedance'].shift(1) == 0) & (data_test['VaR_exceedance'] == 0)).astype(int)
        sum_T_0_1 = data_test['T_0_1'].sum()
        sum_T_1_0 = data_test['T_1_0'].sum()
        sum_T_1_1 = data_test['T_1_1'].sum()
        sum_T_0_0 = data_test['T_0_0'].sum()
        Pi_0 = sum_T_0_1 / (sum_T_0_0 + sum_T_0_1)
        Pi_1 = sum_T_1_1 / (sum_T_1_0 + sum_T_1_1)
        Pi = n_exceed / len(data_test)
        LRind = -2 * np.log(((1 - Pi) ** (sum_T_0_0 + sum_T_1_0)) * (Pi ** (sum_T_0_1 + sum_T_1_1))) + \
            2 * np.log(((1 - Pi_0) ** sum_T_0_0) * (Pi_0 ** sum_T_0_1) * ((1 - Pi_1) ** sum_T_0_1) * (Pi_1 ** sum_T_1_1))
        p_value_ind = 1 - chi2.cdf(LRind, df=1)

        return (p_value_uc > significance_test) and (p_value_ind > significance_test), n_exceed


    def adaptive_backtesting(self, data_train, data_test, window_size=30, max_no_exce=252, alpha = 0.99):
        """
        Implémente un protocole de backtesting adaptatif avec recalibrage.
        
        - data_train : données historiques pour l'entraînement du modèle.
        - data_test : données de test pour le backtest.
        - window_size : taille de la fenêtre d'entraînement.
        - max_no_exce : nombre maximal de jours consécutifs sans exception.

        Retourne :
        - result_var : VaR recalculées après chaque recalibrage.
        - result_date_recalib : Dates des recalibrages effectués.
        - jours_recalibrage : Indices des jours où un recalibrage a eu lieu.
        """
        recalibration_count = 0
        days_without_exception = 0
        result_var = []
        result_date_recalib = []
        jours_recalibrage = []

        while len(data_test) > window_size:
            # Calculer la VaR gaussienne sur la période d'entraînement
            Z_gaussian = self.Var_param_gaussian(data_train["return"], alpha)
            res = self.Var_Hist(Z_gaussian[["return"]], alpha)
            var_99 = res["VaR"]
            for i in range(window_size, len(data_test)):
                subset_test = data_test.iloc[:i]
                result_backtest, n_exceed = self.perform_backtest(subset_test, var_99)

                if not result_backtest:
                    # Si le backtest échoue, procéder au recalibrage
                    days_without_exception = 0
                    jours_recalibrage.append(i)
                    recalibration_count += 1
                    print(f"La VaR est recalibrée à la date {data_test.index[i].strftime('%Y-%m-%d')}\n"
                    f"après {i} jours.")
                    print('-'*60)
                    data_train = pd.concat([data_train.iloc[i:], data_test.iloc[:i]])
                    
                    Z_gaussian = self.Var_param_gaussian(data_train["return"], alpha)
                    res = self.Var_Hist(Z_gaussian[["return"]], alpha)
                    var_99 = res["VaR"]
                    
                    result_var.append(var_99)
                    result_date_recalib.append(data_test.index[i].strftime('%Y-%m-%d'))
                    data_test = data_test.iloc[i:] 
                    
                    break

                ## On compte le nombre de jours consécutifs sans exception      
                if n_exceed == 0:
                    days_without_exception += 1
                else:
                    days_without_exception = 0  # On reset si une exception est détectée

                ## Au cas où on a fait 252 jours consécutifs sans exception et que les tests sont passés, on fait un le recalibrage    
                if days_without_exception >= max_no_exce:
                    recalibration_count += 1
                    days_without_exception = 0  # Reset du compteur
                    data_train = pd.concat([data_train.iloc[max_no_exce:], data_test.iloc[:max_no_exce]])
                    data_test = data_test.iloc[max_no_exce:]
                    break
            else:
                # Si tout est validé, on arrêt le backtest
                break
        return result_var, result_date_recalib, jours_recalibrage
    
    
    #Student VaR
    
    def Var_param_student(self, data, alpha):
        """Calculate VaR using Skewed Student's t-distribution."""
        theta = optimize_parameters(data)
        mu, sigma, gamma, nu = theta
        Z = skew_student_sim(mu, sigma, gamma, nu, len(data))
        Z = pd.DataFrame(Z, index=data.index, columns=["return"])
        return Z
        
    def qqplot(self, df_observed, df_simulated, label):
        """Generate a QQ plot comparing observed and simulated data."""
        quantiles_x = np.percentile(df_observed, np.linspace(0, 100, len(df_observed)))
        quantiles_y = np.percentile(df_simulated, np.linspace(0, 100, len(df_simulated)))

        # Create Plotly figure
        fig = go.Figure()

        # Scatter plot for the observed vs simulated quantiles
        fig.add_trace(go.Scatter(
            x=quantiles_x,
            y=quantiles_y,
            mode='markers',
            name='Observed vs Simulated',
            marker=dict(color='blue', size=8, opacity=0.5)
        ))

        # Add the y=x line (red dashed line)
        fig.add_trace(go.Scatter(
            x=[min(quantiles_x), max(quantiles_x)],
            y=[min(quantiles_x), max(quantiles_x)],
            mode='lines',
            name='y = x',
            line=dict(color='red', dash='dash')
        ))

        # Customize layout
        fig.update_layout(
            title=f"QQ Plot Comparant l'ajustement des données à une {label} ",
            xaxis_title='Empirical Quantiles',
            yaxis_title='Theoretical Quantiles',
            showlegend=True,
            template='plotly_dark',  # Optional, you can customize templates
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Make background transparent
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        return fig
    
    def density_comparison_plot(self, data_train, Z_gaussian, Z_student):
        """
        Generate a density comparison plot using Plotly.

        Parameters:
        - data_train: DataFrame containing historical returns.
        - Z_gaussian: DataFrame with Gaussian simulated returns.
        - Z_student: DataFrame with Student simulated returns.

        Returns:
        - A Plotly figure object.
        """
        # Define a common range for KDE estimation
        x_vals = np.linspace(
            min(data_train["return"].min(), Z_gaussian["return"].min(), Z_student["return"].min()),
            max(data_train["return"].max(), Z_gaussian["return"].max(), Z_student["return"].max()),
            200
        )

        # Compute Kernel Density Estimates (KDEs)
        empirical_kde = gaussian_kde(data_train["return"].dropna())(x_vals)
        gaussian_kde_vals = gaussian_kde(Z_gaussian["return"].dropna())(x_vals)
        student_kde_vals = gaussian_kde(Z_student["return"].dropna())(x_vals)

        # Create Plotly traces
        trace_empirical = go.Scatter(x=x_vals, y=empirical_kde, mode="lines", name="Empirical", line=dict(color="blue"))
        trace_gaussian = go.Scatter(x=x_vals, y=gaussian_kde_vals, mode="lines", name="Gaussian", line=dict(color="red", dash="dash"))
        trace_student = go.Scatter(x=x_vals, y=student_kde_vals, mode="lines", name="Student", line=dict(color="green", dash="dot"))

        # Create the figure
        fig = go.Figure([trace_empirical, trace_gaussian, trace_student])

        # Update layout
        fig.update_layout(
            title="Density Comparison: Gaussian vs Student vs Empirical",
            xaxis_title="Returns",
            yaxis_title="Density",
            template="plotly_white",
            showlegend=True
        )

        return fig
        
    def exceedance_test(self, data, VaR, alpha_exceed=0.05):
        """Test for exceedances of VaR and calculate confidence intervals."""
        data["exceed_VaR"] = (data.loc[:, "return"] < VaR).astype(int)
        num_exceed = data["exceed_VaR"].sum()
        
        p_hat = num_exceed / len(data)
        z = st.norm.ppf(1 - alpha_exceed / 2)
        margin = z * np.sqrt(p_hat * (1 - p_hat) / len(data))
        return (p_hat - margin, p_hat + margin)

    # VaR GEV
    # 1. Déterminer une taille de bloc s et construire un échantillon de maxima
    def block_maxima(self, data, block_size):
        """
        Calcule les maxima par bloc pour une série donnée.
        
        Parameters:
        - data: Série temporelle des pertes.
        - block_size: Taille du bloc (en nombre d'observations).
        
        Returns:
        - block_max: Liste des maxima par bloc.
        """
        n = len(data)
        block_max = [max(data[i:i + block_size]) for i in range(0, n, block_size)]
        return np.array(block_max)


    ## Estimer les paramètres de la loi Gumbel
    def fit_gumbel(self, data):
        """
        Estime les paramètres de la loi GEV par maximum de vraisemblance.
        
        Parameters:
        - data: Série des maxima par bloc.
        
        Returns:
        - shape (ξ), location (μ), scale (σ).
        """
        def neg_log_likelihood(params):
            loc, scale = params
            if scale <= 0:
                return np.inf
            return -np.sum(gumbel_r.logpdf(data, loc=loc, scale=scale))
        
        # Estimation initiale
        initial_guess = [np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        loc, scale = result.x
        return loc, scale, -neg_log_likelihood([loc, scale])

    def gumbel_plot(self, data, loc, scale):
        """
        Trace le Gumbel plot pour vérifier l'hypothèse ξ=0.
        
        Parameters:
        - data: Série des maxima par bloc.
        - loc, scale: Paramètres de la loi Gumbel.
        """
        # Calculate the theoretical and empirical quantiles
        theoretical_quantiles = gumbel_r.ppf(np.linspace(0.01, 0.99, 100), loc, scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))

        # Create the Gumbel QQ-Plot using Plotly
        fig = go.Figure()

        # Scatter plot for the data points
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode='markers',
            name='Empirical vs Theoretical',
            marker=dict(color='blue', size=8)
        ))

        # Add the line y=x (red dashed line) for reference
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            name='y = x',
            line=dict(color='red', dash='dash')
        ))

        # Layout customization
        fig.update_layout(
            title='QQ-Plot (validation de la loi Gumbel ex-ante)',
            xaxis_title='Quantiles théoriques (Gumbel)',
            yaxis_title='Quantiles empiriques',
            showlegend=True,
            template='plotly_dark',  # Optional: You can choose other templates
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Make background transparent
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        return fig

    ## Estimer les paramètres de la loi GEV
    def fit_gev(self, data):
        """
        Estime les paramètres de la loi GEV par maximum de vraisemblance.
        
        Parameters:
        - data: Série des maxima par bloc.
        
        Returns:
        - shape (ξ), location (μ), scale (σ).
        """
        def neg_log_likelihood(params):
            shape, loc, scale = params
            if scale <= 0:
                return np.inf
            return -np.sum(genextreme.logpdf(data, shape, loc=loc, scale=scale))
        
        # Estimation initiale
        initial_guess = [0.1, np.mean(data), np.std(data)]
        result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
        shape, loc, scale = result.x
        return shape, loc, scale, -neg_log_likelihood([shape, loc, scale])

    def LR_test(self, logL1, logL2):
        LRT_stat = 2 * (logL2 - logL1)  # Model 2 vs Model 1
        p_value = 1 -stats.chi2.cdf(LRT_stat, 1)

        print(f"Likelihood Ratio Statistic: {LRT_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("GEV model significantly improves the fit over Gumbel model.")
            return False
        else:
            print("No significant improvement; prefer the Gumbel model.")
            return True

    # 4. Validation ex-ante (QQ-plot, etc.)
    def gev_plot(self, data, shape, loc, scale):
        """
        Valide l'ajustement de la loi GEV par QQ-plot.
        
        Parameters:
        - data: Série des maxima par bloc.
        - shape, loc, scale: Paramètres de la loi GEV.
        """
        # Calculate the theoretical and empirical quantiles
        theoretical_quantiles = genextreme.ppf(np.linspace(0.01, 0.99, 100), shape, loc, scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))

        # Create the QQ-Plot using Plotly
        fig = go.Figure()

        # Scatter plot for the data points
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode='markers',
            name='Ajustement à une GEV',
            marker=dict(color='blue', size=8)
        ))

        # Add the line y=x (red dashed line) for reference
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            name='y = x',
            line=dict(color='red', dash='dash')
        ))

        # Layout customization
        fig.update_layout(
            title='QQ-Plot (validation de la loi GEV ex-ante)',
            xaxis_title='Quantiles théoriques (GEV)',
            yaxis_title='Quantiles empiriques',
            showlegend=True,
            template='plotly_dark',  # Optional: You can choose other templates
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Make background transparent
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        return fig

    # 5. Calculer la VaR TVE par MB pour alpha = 99%
    def calculate_var_gve(self, data, block_size, alpha=0.99):
        """
        Calcule la VaR TVE pour un niveau de confiance donné.
        
        Parameters:
        - shape, loc, scale: Paramètres de la loi GEV.
        - alpha: Niveau de confiance (par défaut 99%).
        
        Returns:
        - VaR TVE.
        """
        block_max = self.block_maxima(data, block_size)

        # 2. Estimer la Gumbel
        _, _, logL1 = self.fit_gumbel(block_max)

        # 3. Estimer les paramètres de la loi GEV
        _, _, _, logL2 = self.fit_gev(block_max)

        # Compare Gumbel and GEV
        if self.LR_test(logL1, logL2):
            loc, scale, _ = self.fit_gumbel(block_max)
            VaR = gumbel_r.ppf(alpha**block_size, loc=loc, scale=scale)
            fig = self.gumbel_plot(block_max, loc, scale)
            return VaR, fig
        else:
            shape, loc, scale, logL2 = self.fit_gev(block_max)
            VaR = genextreme.ppf(alpha**block_size, shape, loc=loc, scale=scale)
            fig = self.gev_plot(block_max, shape, loc, scale)
            return VaR, fig
        
    # VaR GPD
    def mean_excess_plot(self, data, u_min=0, u_max=None, step=0.01):
        """
        Trace le Mean Excess Plot pour déterminer un seuil u approprié.

        Parameters:
        - data: Série des pertes (rendements négatifs).
        - u_min: Seuil minimal à considérer.
        - u_max: Seuil maximal à considérer.
        - step: Pas pour l'incrémentation des seuils.

        Returns:
        - Un graphique du Mean Excess Plot sous forme de figure Plotly.
        """
        if u_max is None:
            u_max = np.quantile(data, 0.99)  # Ne pas considérer les valeurs trop extrêmes

        thresholds = np.arange(u_min, u_max, step)
        mean_excess = [np.mean(data[data > u] - u) for u in thresholds]

        # Create the Mean Excess Plot using Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=thresholds,
            y=mean_excess,
            mode='markers+lines',
            name='Mean Excess',
            marker=dict(color='blue', size=8)
        ))

        # Add zero line for reference
        fig.add_trace(go.Scatter(
            x=[thresholds[0], thresholds[-1]],
            y=[0, 0],
            mode='lines',
            name='Zero Line',
            line=dict(color='red', dash='dash')
        ))

        # Layout customization
        fig.update_layout(
            title='Mean Excess Plot',
            xaxis_title='Seuil u',
            yaxis_title='Moyenne des excès',
            showlegend=True,
            template='plotly_dark',  # Optional: You can choose other templates
            plot_bgcolor='rgba(0, 0, 0, 0)',  # Make background transparent
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        return fig

    
    
    def fit_gpd(self, data, u):
        """
        Ajuste une loi GPD aux excès au-dessus du seuil u.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.

        Returns:
        - Paramètres de la GPD (shape, scale).
        """
        excess = data[data > u] - u
        params = genpareto.fit(excess, floc=0)  # Ajustement de la GPD
        return params


    def gpd_validation(self, data, u, shape, scale):
        """
        Validation ex-ante de l'ajustement de la GPD.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.
        - shape, scale: Paramètres de la GPD.

        Returns:
        - QQ-plot et PP-plot sous forme de figure Plotly.
        """
        excess = data[data > u] - u
        n = len(excess)
        theoretical_quantiles = genpareto.ppf(np.linspace(0, 1, n), shape, loc=0, scale=scale)
        empirical_quantiles = np.sort(excess)

        # QQ-Plot
        qq_trace = go.Scatter(
            x=theoretical_quantiles,
            y=empirical_quantiles,
            mode='markers',
            name='Empirical vs Theoretical',
            marker=dict(color='blue')
        )

        qq_line = go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            name='Ideal Line',
            line=dict(color='red', dash='dash')
        )

        # PP-Plot
        theoretical_probs = genpareto.cdf(empirical_quantiles, shape, loc=0, scale=scale)
        empirical_probs = np.linspace(0, 1, n)

        pp_trace = go.Scatter(
            x=theoretical_probs,
            y=empirical_probs,
            mode='markers',
            name='Ajustement à une Gumbel',
            marker=dict(color='blue')
        )

        pp_line = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Ideal Line',
            line=dict(color='red', dash='dash')
        )

        # Creating the figure
        fig = go.Figure()

        # Add QQ plot data
        fig.add_trace(qq_trace)
        fig.add_trace(qq_line)

        # Add PP plot data
        fig.add_trace(pp_trace)
        fig.add_trace(pp_line)

        # Update the layout to include titles and axis labels
        fig.update_layout(
            title='Validation of GPD Fit: QQ-Plot and PP-Plot',
            xaxis_title='Theoretical Quantiles / Probabilities',
            yaxis_title='Empirical Quantiles / Probabilities',
            template='plotly_dark',  # Optional, adjust the template as needed
            showlegend=True,
            xaxis=dict(
                title='Quantiles (QQ-Plot) / Probabilities (PP-Plot)',
            ),
            yaxis=dict(
                title='Quantiles (QQ-Plot) / Probabilities (PP-Plot)',
            ),
            legend=dict(
                x=0.1, y=0.9,
                traceorder='normal',
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='Black',
                borderwidth=1
            ),
        )

        return fig

    def var_tve_pot(self, data, u, shape, scale, alpha=0.99):
        """
        Calcule la VaR TVE par l'approche PoT.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.
        - shape, scale: Paramètres de la GPD.
        - alpha: Niveau de confiance (par défaut 99%).

        Returns:
        - VaR TVE.
        """
        n = len(data)
        nu = len(data[data > u])  # Nombre d'excès
        var = u + (scale / shape) * (((n / nu) * (1 - alpha)) ** (-shape) - 1)
        return var

    def calibrate_u(self, data, alpha=0.99, step=0.0001):
        """
        Automatically calibrates the threshold u for Peak Over Threshold (PoT).
        
        Parameters:
        - data: Loss data.
        - alpha: Confidence level.
        - u_min, u_max: Range of u values.
        - step: Step size for threshold selection.
        
        Returns:
        - Optimal u value.
        """
        u_min = np.quantile(data, 0.90) # To avoid recurrent values
        u_max = np.quantile(data, 0.99)  # Avoid extreme values
        
        thresholds = np.arange(u_min, u_max, step)
        shapes = []
        scales = []
        var_tve_values = []

        for u in thresholds:
            excess = data[data > u] - u
            if len(excess) > 10:  # Ensure enough exceedances
                shape, loc, scale = self.fit_gpd(data, u)
                shapes.append(shape)
                scales.append(scale)
                var_tve_values.append(self.var_tve_pot(data, u, shape, scale, alpha))

        # Identify the most stable threshold u
        shape_stability = np.abs(np.diff(shapes))
        scale_stability = np.abs(np.diff(scales))

        stability = shape_stability + scale_stability
        u_optimal_idx = np.argmin(stability) + 1  # Add 1 to match index
        #plt.plot(thresholds[:-1],stability)  # Visualize how stability moves with thresholds
        u_optimal = thresholds[u_optimal_idx]

        return u_optimal

    # VAR DYNAMIQUE

    def dynamic_VaR(self, data_train, data_test, alpha, start_test):
        """
        Calculate dynamic VaR and plot the results with a vertical line at the start of the test data.
    
        Parameters:
            data_train (pd.DataFrame): Training data containing returns.
            data_test (pd.DataFrame): Test data containing returns.
            alpha (float): Confidence level for VaR (e.g., 0.05 for 95% confidence).
            start_test (str or datetime): Date indicating the start of the test data.
    
        Returns:
            fig: Plotly figure object.
        """
        # Fitting AR(1)_GARCH(1,1)
        combined_model = arch_model(data_train['return'], mean='AR', lags=1, vol='Garch', p=1, q=1)
        combined_fit = combined_model.fit()
    
        # Extract standardized residuals
        std_residuals = combined_fit.std_resid.dropna().to_numpy()
        std_residuals = -std_residuals  # Invert residuals
    
        # VaR on the standard residuals using GPD
        u = 5e-4
        #u = self.calibrate_u(std_residuals, alpha)  # Calibrate optimal threshold
        shape, loc, scale = self.fit_gpd(std_residuals, u)  # Fit GPD
        VaR_res = -self.var_tve_pot(std_residuals, u, shape, scale, alpha)  # Calculate VaR
    
        # Extracting estimated parameters
        mu, phi, omega, a, b = combined_fit.params
    
        # Calculate dynamic VaR on all data (train + test)
        data = pd.concat([data_train, data_test])  # Combine train and test data
    
        # Initialize mu and vol
        data["mu"] = mu + phi * data["return"].shift()
        data["mu"].iloc[0] = mu  # Set initial value
    
        data["vol"] = np.sqrt(omega / (1 - a - b))  # Initialize volatility
    
        # Update volatility dynamically
        for t in range(1, len(data)):
            data["vol"].iloc[t] = np.sqrt(
                omega
                + a * (data["return"].iloc[t - 1] - data["mu"].iloc[t - 1]) ** 2
                + b * data["vol"].iloc[t - 1] ** 2
            )
    
        # Calculate dynamic VaR
        data["VaR"] = data["mu"] + data["vol"] * VaR_res
    
        # Create a Plotly figure
        fig = go.Figure()
    
        # Add VaR line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["VaR"],
                mode="lines",
                name="VaR",
                line=dict(color="red", dash="dash"),
            )
        )
    
        # Add returns line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["return"],
                mode="lines",
                name="Rendements",
                line=dict(color="blue"),
            )
        )
    
        # Identify points where VaR exceeds returns
        exceedance_points = data[data["VaR"] > data["return"]]
    
        # Add exceedance points
        fig.add_trace(
            go.Scatter(
                x=exceedance_points.index,
                y=exceedance_points["return"],
                mode="markers",
                name="Exceptions",
                marker=dict(color="red", size=8),
            )
        )
    
    
        # Add annotations for exceedance points
        for date, return_value in exceedance_points["return"].items():
            fig.add_annotation(
                x=date,
                y=return_value,
                #text=date.strftime('%Y-%m-%d'),
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
            )
    
        # Update layout
        fig.update_layout(
            title="Dynamic VaR vs Rendements",
            xaxis_title="Date",
            yaxis_title="Valeur",
            legend_title="Legend",
            hovermode="x unified",
        )
    
        # Return the figure
        return fig
        
    
    # Fitting VaR
    def fit(self, start_train, start_test, end_test, alpha):
        """Fit the model and calculate VaR and ES using different methods."""
        # Load data
        self.load_data()
        self.plot_data()
        
        # Train/Test split
        data_train, data_test = self.train_test_split(start_train=start_train, start_test=start_test, end_test=end_test)
        
        # Stats
        summary = {"Train set ":data_train.describe(), "Test set":data_test.describe()}
        
        # Historical VaR and ES
        res = self.Var_Hist(data_train[["return"]], alpha)
        VaR_hist, ES_hist = res["VaR"], res["ES"]
        bin_IC = self.exceedance_test(data_test[["return"]], VaR_hist, alpha_exceed=0.05)
        
        # Bootsrap historical VaR with CI
        res = self.Var_Hist_Bootstrap(data_train[["return"]], alpha, B = 252, alpha_IC = 0.90, M = 500)
        VaR_bootstrap = res["VaR"]
        VaR_IC = res
        
        # Gaussian parametric VaR and ES
        Z_gaussian = self.Var_param_gaussian(data_train["return"], alpha)
        res = self.Var_Hist(Z_gaussian[["return"]], alpha)
        VaR_gaussian, ES_gaussian = res["VaR"], res["ES"]
        VaR_gaussian_10_day = np.sqrt(10) * VaR_gaussian  # Corrected 10-day VaR calculation
        qqplot_gaussian = self.qqplot(data_train["return"].values, Z_gaussian["return"].values, label="Gaussienne")

        ## VaR at 10 days horizon 
        S0 = data_train['Close'].iloc[-1]
        mu = np.mean(data_train['return'])
        sigma = np.std(data_train['return'])
        t = 11
        num_simulations = 1000
        St = self.simulate_price_paths(t, S0, mu, sigma, num_simulations)
        # Calcul des rendements log
        log_returns = self.calculate_log_returns(St, S0)
        # Calcul de la VaR à 99%
        VaR_gaussian_10_day_diff = self.calculate_var(log_returns)
        
        # Student parametric VaR and ES
        Z_student = self.Var_param_student(data_train["return"], alpha)
        res = self.Var_Hist(Z_student[["return"]], alpha)
        VaR_student, ES_student = res["VaR"], res["ES"]
        qqplot_student = self.qqplot(data_train["return"].values, Z_student["return"].values, label ="Student")
        
        # Comparing Gaussian and Student calibrations
        fig = plt.figure()
        sns.kdeplot(Z_gaussian["return"], label="Gaussian")
        sns.kdeplot(Z_student["return"], label="Student")
        sns.kdeplot(data_train["return"], label="Empirical")
        plt.title('Density Comparison: Gaussian vs Student vs Empirical')
        plt.legend()
        
        # Back testing
        
        res = self.adaptive_backtesting(data_train, data_test, window_size=30, max_no_exce=252, alpha = 0.99)
        res = {"Days since last recalibration":res[0], "recalibrated VaR":res[1], "Recalibration date":res[2]}
        
        # VaR GEV
        
        block_size = 20  # Taille de bloc (max mensuel)
        block_max = self.block_maxima(-data_train["return"].to_numpy(), block_size)

        ## 2. Tracer le Gumbel plot
        loc, scale, _ = self.fit_gumbel(block_max)
        qqplot_gumbel = self.gumbel_plot(block_max, loc, scale)
        

        # VaR GPD
        mrlplot = self.mean_excess_plot(-data_train["return"].to_numpy(), u_min=0, step=0.001)
        u = self.calibrate_u(-data_train["return"].to_numpy(), alpha)  ## Calibrate optimal u
        shape, loc, scale =self.fit_gpd(-data_train["return"].to_numpy(), u)
        VaR_gpd = - self.var_tve_pot(-data_train["return"].to_numpy(), u, shape, scale, alpha)
        qqplot_gpd = self.gpd_validation(-data_train["return"].to_numpy(), u, shape, scale)
        
        return {
            "stats": summary,
            "VaR_hist": VaR_hist,
            "VaR_bootstrap":VaR_bootstrap,
            "VaR_IC":VaR_IC,
            "ES_hist": ES_hist,
            "VaR_gaussian": VaR_gaussian,
            "VaR_gaussian_10_day_diff": VaR_gaussian_10_day_diff,
            "VaR_gaussian_10_day": VaR_gaussian_10_day,
            "ES_gaussian": ES_gaussian,
            "VaR_student": VaR_student,
            "ES_student": ES_student,
            "qqplot_gaussian": qqplot_gaussian,
            "qqplot_student": qqplot_student,
            "Gaussian vs Student calibrations":fig,
            "mrlplot": mrlplot,
            "VaR_gpd": VaR_gpd,
            "qqplot_gpd": qqplot_gpd,
            "back_test":res
        }