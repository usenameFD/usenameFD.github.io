import numpy as np
import yfinance as yf
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis,chi2, norm,ks_1samp,genextreme, kstest,gumbel_r, genpareto, shapiro, jarque_bera
import seaborn as sns
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.graph_objects as go



#########################################################################
############ Fonctions pour les statistiques descriptives ###############
#########################################################################


def descriptive_statistics(df, columns, alpha=0.05):
    """
    Calcule les statistiques descriptives pour une liste de colonnes d'un DataFrame, ainsi que les tests de stationnarité ADF et KPSS.
    Renvoie également si chaque série est stationnaire ou non.

    Paramètres :
    - df (pd.DataFrame) : Le DataFrame contenant les données.
    - columns (list) : La liste des noms de colonnes pour lesquelles calculer les statistiques.
    - alpha (float) : Le seuil de signification pour les tests ADF et KPSS (par défaut 0.05).

    Retourne :
    - stats_df (pd.DataFrame) : Un DataFrame contenant les statistiques descriptives et la stationnarité pour chaque colonne 
    """
    stats = {'Statistic': ['Mean', 'Median', 'Variance', 'Standard Deviation', 'Skewness', 'Kurtosis', 'ADF p-value', 'KPSS p-value', 'ADF Stationarity', 'KPSS Stationarity']}

    for col in columns:
        # Test ADF
        adf_pvalue = adfuller(df[col])[1]
        if adf_pvalue < alpha:
            adf_stationary = 'Stationnaire'
        else:
            adf_stationary = 'Non stationnaire'
        
        # Test KPSS
        kpss_pvalue = kpss(df[col], nlags='auto')[1]
        if kpss_pvalue < alpha:
            kpss_stationary = 'Non stationnaire'
        else:
            kpss_stationary = 'Stationnaire'
        
        stats[col] = [
            df[col].mean(),
            df[col].median(),
            df[col].var(),
            df[col].std(),
            skew(df[col]),
            kurtosis(df[col]),
            adf_pvalue,
            kpss_pvalue,
            adf_stationary,
            kpss_stationary
        ]

    stats_df = pd.DataFrame(stats)
    return stats_df




import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, columns):
    """
    Affiche les histogrammes des distributions pour chaque colonne spécifiée avec la courbe de densité.
    
    Paramètres :
    - df (pd.DataFrame) : Le DataFrame contenant les données.
    - columns (list) : La liste des noms de colonnes à tracer.
    
    Retourne :
    - None : Affiche directement les graphiques.
    """
    n = len(columns)
    plt.figure(figsize=(6 * n, 5))

    for i, col in enumerate(columns, 1):
        plt.subplot(1, n, i)
        # Histogramme
        plt.hist(df[col], bins=20, color='skyblue', edgecolor='black', density=True)
        # Courbe de densité
        sns.kdeplot(df[col], color='red', linewidth=2)
        plt.title(f'Distribution de {col}')
        plt.xlabel('Valeurs')
        plt.ylabel('Fréquence')

    plt.suptitle("Distributions des colonnes de l'échantillon")
    plt.tight_layout()
    plt.show()




######################################################################################
################### Fonctions pour le calcul des différentes VaR #####################
######################################################################################


#### VaR Historique 

def VaR_Hist(returns, confidence_level=0.99):
    """
    Calcule la Value at Risk (VaR) historique sur un horizon de 1 jour.

    Paramètres :
    - returns : Série de rendements (log-returns ou simples)
    - confidence_level : Niveau de confiance (99% par défaut)

    Retourne :
    - La VaR historique au seuil donné
    """
    var_threshold = np.percentile(returns, 100 * (1 - confidence_level))
    return var_threshold




### VaR Bootstrap

def VaR_Hist_Bootstrap(returns, confidence_level=0.99, n_iterations=10000, ci_level=0.90):
    """
    Calcule la VaR Bootstrap avec un intervalle de confiance via la méthode des quantiles bootstrap.

    Paramètres :
    - returns : Série des log-returns historiques
    - confidence_level : Niveau de confiance pour la VaR (99% par défaut)
    - n_iterations : Nombre d'itérations bootstrap (par défaut 1000)
    - ci_level : Niveau de confiance pour l'intervalle (95% par défaut)

    Retourne :
    - La VaR bootstrap au seuil spécifié
    - L'intervalle de confiance pour cette VaR
    """
    # Génération des échantillons bootstrap et calcul des VaR associée
    bootstrap_samples = np.random.choice(returns, size=(n_iterations, len(returns)), replace=True)
    var_bootstrap_samples = np.percentile(bootstrap_samples, 100 * (1 - confidence_level), axis=1)

    # Estimation de la VaR bootstrapée (médiane des estimations bootstrap)
    var_estimate = np.median(var_bootstrap_samples)

    # Calcul de l'intervalle de confiance
    lower_bound = np.percentile(var_bootstrap_samples, (1 - ci_level) / 2 * 100)
    upper_bound = np.percentile(var_bootstrap_samples, (1 + ci_level) / 2 * 100)

    return var_estimate, (lower_bound, upper_bound)



################ Fonction pour le calcul du nombre d'exceptions #################################
#################################################################################################

def var_exceptions(test_data, var_threshold):
    """
    Cette fonction calcule et affiche le nombre d'exceptions par rapport à la VaR historique à un seuil donné. 
    Elle identifie les périodes où les rendements log sont inférieurs à la VaR à 99% et calcule le pourcentage d'exceptions.

    Paramètres :
    - test_data (pd.DataFrame) : Un DataFrame contenant les données de test
    - var_threshold (float) : La valeur de la VaR à 99% 

    Retourne :
    - None : La fonction affiche directement le nombre d'exceptions et le pourcentage d'exceptions par rapport à la taille de l'échantillon.
    """
    
    test_data['Exception'] = test_data['Log Return'] < var_threshold
    n_exceptions = test_data['Exception'].sum()

    # Afficher les résultats
    print(f"Nombre d'exceptions (log-returns inférieurs à la VaR à 99%) : {n_exceptions}")
    print(f"Pourcentage d'exceptions : {n_exceptions / len(test_data) * 100:.2f}%")




################# Test d'unconditional coverage pour verifier si le nombre d'exception ##############
################  observé est cohérent avec le seuil de la VaR fixé #################################


def unconditional_coverage_test(data_test, var_99, significance_level=0.01):
    """
    Effectue le test d'unconditional coverage pour vérifier si la VaR couvre correctement
    les exceptions sur un échantillon de test.

    Paramètres :
    - data_test : DataFrame contenant les log-returns de test avec une colonne 'Log Return'
    - var_99 : Valeur de la VaR à 99% (le seuil pour les dépassements)
    - significance_level : Niveau de significativité du test (par défaut 0.01 pour 99% de confiance)

    Retourne :
    - La statistique du test LR
    - La p-value du test
    """
    # Calcul des dépassements et du nombre d'exceptions observées
    data_test['VaR_exceedance'] = data_test['Log Return'] < var_99  # Vérifier si le log-return est inférieur à la VaR
    n = data_test['VaR_exceedance'].sum()  

    # Calcul de la probabilité de dépassement empirique
    obs = len(data_test)
    prob_dep_em = n / obs

    # Calcul de la statistique LR du test d'unconditional coverage et de la p-value associée
    LR_uc = -2 * np.log(((1 - significance_level) ** (obs - n)) * (significance_level ** n)) + \
            2 * np.log(((1 - prob_dep_em) ** (obs - n)) * (prob_dep_em ** n))

    p_value = 1 - chi2.cdf(LR_uc, df=1)


    print(f"Nombre d'exceptions observées : {n}")
    print('-'*50)
    print(f"Nombre total d'observations : {obs}")
    print('-'*50)
    print(f"Probabilité de dépassement empirique : {prob_dep_em:.4f}")
    print('-'*50)
    print(f"Statistique LR pour le test d'unconditional coverage : {LR_uc:.4f}")
    print('-'*50)
    print(f"P-value du test d'unconditional coverage : {p_value:.4f}")
    print('-'*50)

    # Interprétation de la p-value
    if p_value < 0.05:
        print("L'hypothèse nulle est rejetée : Le modèle de VaR ne couvre pas correctement les exceptions.")
    else:
        print("L'hypothèse nulle n'est pas rejetée : Le modèle de VaR semble couvrir correctement les exceptions.")

    return LR_uc, p_value




########################################################################################################
###########     Fonction calculant la VaR gaussienne d’un ensemble de log-rendements ###################

def VaR_Gauss(data_train, alpha=0.99):
    """
    Calcule la VaR gaussienne à partir des rendements log de l'échantillon d'entraînement.
    
    Paramètres :
    - data_train (pd.DataFrame) : DataFrame avec les rendements log ('Log Return').
    - alpha (float) : Niveau de confiance (par défaut 0.99).
    
    Retourne :
    - var_99 (float) : VaR à 99% calculée selon la méthode gaussienne.
    """
    mu = data_train['Log Return'].mean() 
    sigma = data_train['Log Return'].std()  
    var_99 = mu - norm.ppf(alpha) * sigma  
    
    return var_99



#####################################################################################################
######################## Outils de validations ex-anté de la VaR Gaussienne #########################

from scipy.stats import norm

def plot_kde_vs_gauss(data):
    r = data['Log Return']
    mu, sigma = r.mean(), r.std()
    
    sns.kdeplot(r, label='Densité empirique', color='blue')
    
    x = np.linspace(r.min(), r.max(), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), label='Densité normale', color='red')
    
    plt.axvline(mu - norm.ppf(0.99) * sigma, color='black', linestyle='--', label='VaR Gaussienne 99%')
    plt.legend()
    plt.title("densité empirique vs densité normale")
    plt.show()



#################################  QQ-plot  #################################################

def plot_qq(data):
    """
    Affiche un QQ-plot pour comparer la distribution des données avec une loi normale.

    Paramètres :
    - data : Série de données à analyser 
    """
    plt.figure(figsize=(8, 6))
    stats.probplot(data.dropna(), dist="norm", plot=plt)
    plt.title("QQ-Plot des Log-Returns")
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Quantiles des données")
    plt.grid()
    plt.show()


#################### Test d'adéquation de Kolmogorov- Smirnov ###################################

def ks_test(data):
    """
    Effectue un test de Kolmogorov-Smirnov pour vérifier si les données suivent une loi normale.

    Paramètres :
    - data : Série de données à tester (ex: log-returns)

    Retourne :
    - La statistique KS et la p-valeur
    """
    # Estimation des paramètres de la loi normale
    mean, std = data.mean(), data.std()  
    ks_stat, p_value = ks_1samp(data, lambda x: norm.cdf(x, loc=mean, scale=std))

    print(f"Statistique KS: {ks_stat:.4f}")
    print(f"P-valeur: {p_value:.4f}")

    if p_value < 0.05:
        print("Resultat du test : Rejet de l'hypothèse de normalité au seuil de 5% ")
    else:
        print("Resultat du test : On ne rejette pas l'hypothèse de normalité au seuil de 5% ")



################# Calcul de la VaR Gaussienne à horizon 10 jours par scaling ######################

def var_gauss_horizon(var_gauss, horizon):
    """
    Calcule la VaR gaussienne à un horizon donné en appliquant la méthode du scaling.
    
    Paramètres :
    - var_gauss (float) : VaR gaussienne quotidienne à un niveau de confiance donné.
    - horizon (int) : Nombre de jours (horizon) sur lequel on souhaite calculer la VaR.
    
    Retourne :
    - VaR à l'horizon donné, multipliée par 100 (en pourcentage).
    """
    var_horizon = np.sqrt(horizon) * var_gauss * 100  
    print(f"VaR à {horizon} jours : {var_horizon:.2f}%")
    
    return var_horizon



############# Calcul de la VaR Gaussienne à 10 jours par la méthode de diffusion d'un actif ###########


def simulate_price_paths(t, S0, mu, sigma, num_simulations):
    """
    Simule les trajectoires de prix d'un actif financier en utilisant un modèle de diffusion géométrique brownien.
    
    Paramètres :
    - t (int) : Le nombre de périodes (jours, mois, etc.) pour chaque trajectoire simulée.
    - S0 (float) : Le prix initial de l'actif (valeur à la période 0).
    - mu (float) : Le rendement moyen (moyenne des log-rendements).
    - sigma (float) : La volatilité (écart-type des log-rendements).
    - num_simulations (int) : Le nombre de trajectoires de prix à simuler.
    
    Retourne :
    - St (numpy.ndarray) : Une matrice de taille (num_simulations, t), où chaque ligne correspond à une trajectoire simulée de prix de l'actif.
    """
    
    t=t+1
    St = np.zeros((num_simulations, t))

    # Simuler les trajectoires de prix
    for i in range(num_simulations):
        
        St[i, 0] = S0
        Z = np.random.normal(0, 1, t)

        for j in range(1, t):
            St[i, j] = St[i, j-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.sqrt(1) * Z[j-1])
    return St



def calculate_log_returns(St, S0):
    """
    Calcule les rendements log-transformés à partir des prix simulés.

    Paramètres :
    - St (numpy.ndarray) : Matrice des prix simulés (dimensions : num_simulations, t).
    - S0 (float) : Prix initial de l'actif à t=0.

    Retourne :
    - numpy.ndarray : Rendements log-transformés à la dernière période pour chaque simulation.
    """
    # Calcul des log-returns
    log_returns = np.log(St[:, -1] / S0)
    
    return log_returns



def calculate_var(log_returns, t=10, confidence_level=0.99):
    """
    Calcule la VaR à partir des rendements log-transformés.

    Paramètres :
    - log_returns (numpy.ndarray) : Tableau des rendements log-transformés.
    - t (int) : Horizon temporel (par défaut 10 jours).
    - confidence_level (float) : Niveau de confiance pour la VaR (par défaut 0.99).

    Retourne :
    - float : La VaR à partir des rendements log-transformés.
    """
    var = np.percentile(log_returns, 100 * (1 - confidence_level))
    
    # Affichage avec l'horizon et le niveau de confiance
    print(f"VaR diffusée à {confidence_level * 100}% pour un horizon de {t} jours : {var:.4f}")
    
    return var



def plot_simulations(St, num_trajectoires, t, title="Simulations des trajectoires des log returns", figsize=(10, 6)):
    """
    Trace les trajectoires simulées des prix d'un actif.
    
    Paramètres :
    - St : ndarray de forme (num_simulations, t) contenant les trajectoires de prix simulées.
    - num_simulations : int, nombre de simulations à tracer.
    - t : int, nombre de jours ou de points de temps (longueur de chaque trajectoire).
    - title : str, titre du graphique.
    - figsize : tuple, taille de la figure du graphique (par défaut (10, 6)).
    """
    plt.figure(figsize=figsize)

    # Générer une palette de couleurs distinctes
    colors = plt.cm.get_cmap('tab10', 10)  

    # Tracer les trajectoires pour chaque simulation
    for i in range(num_trajectoires): 
        plt.plot(St[i, :], label=f'Simulation {i+1}', color=colors(i), alpha=0.7)

    # Ajouter les éléments du graphique
    plt.title(f'{title} sur {t} jours')
    plt.xlabel('Jours')
    plt.ylabel('Prix')
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    plt.show()



######################################################################################################
################# Calcul de la VaR Gaussienne en surpondérant les observations récentes ##############

def calculate_mu_sigma_ewma(log_returns, lambda_):
    """
    Calcule les paramètres de la moyenne (μ̂(λ)) et de la variance (σ̂²(λ)) selon la méthode EWMA
    pour une valeur spécifique de λ.

    Paramètres :
    - log_returns (numpy.ndarray) : Tableau des rendements log-transformés.
    - lambda_ (float) : Valeur de λ pour laquelle on veut calculer les paramètres.

    Retourne :
    -  les valeurs de μ̂(λ) et σ̂²(λ) pour la valeur donnée de λ.
    """
    T = len(log_returns)

    # Calcul des poids EWMA
    weights = np.array([lambda_**i * (1 - lambda_) for i in range(T)])
    normalized_weights = weights / weights.sum()  

    # Calcul de la moyenne et de la variance pondérées des rendements (mu_lambda)
    mu_lambda = np.sum(normalized_weights * log_returns)
    sigma_lambda_squared = np.sum(normalized_weights * (log_returns - mu_lambda)**2)
    sigma_lambda = np.sqrt(sigma_lambda_squared)
    
    # Affichage des résultats
    print(f"Pour λ = {lambda_}:")
    print(f"  - μ̂(λ) = {mu_lambda:.4f}")
    print(f"  - σ̂(λ) = {sigma_lambda:.4f}")
    print("-" * 40)

    return [ mu_lambda,sigma_lambda]



def calculate_var_gauss_ewma(log_returns, lambda_, alpha=0.99):
    """
    Calcule la VaR gaussienne à 1 jour sur la base d'apprentissage en utilisant la méthode EWMA.

    Paramètres :
    - log_returns (numpy.ndarray) : Tableau des rendements log-transformés.
    - lambda_ (float) : Valeur de λ pour la méthode EWMA.
    - alpha (float) : Niveau de confiance pour la VaR (par défaut 0.99).

    Retourne :
    - float : La VaR gaussienne EWMA à 1 jour.
    """
    # Calcul des paramètres μ̂(λ) et σ̂²(λ) 
    mu_lambda, sigma_lambda = calculate_mu_sigma_ewma(log_returns, lambda_)
    # Calcul de la VaR Gaussienne EWMA
    quantile = norm.ppf( alpha)
    var_ewma = mu_lambda - quantile * sigma_lambda

    print(f"VaR gaussienne EWMA à 1 jour (α = {alpha * 100}%): {var_ewma:.4f}")
    print("-"*40)
    
    return var_ewma



####################################################################################################
################################ VaR Skew- Student #################################################


## Fonction de densité
import scipy.stats as st
def f_skew_student(x, mu, sigma, gamma, nu):
    arg = (x - mu) / sigma
    arg2 = gamma * arg * np.sqrt((nu + 1) / (arg**2 + nu))
    f = st.t.pdf(x, df=nu, loc=mu, scale=sigma)
    F = st.t.cdf(arg2, df=nu + 1)
    return 2 * f * F


##Fonction de vraisemblance 

def log_likelihood(theta, x):
    mu, sigma, gamma, nu = theta
    pdf_values = f_skew_student(x, mu, sigma, gamma, nu)
    log_lik = np.sum(np.log(pdf_values))
    return -log_lik


## optimisation des paramètres 


def optimize_parameters(x):
    """
    Optimise les paramètres de la loi de Skew Student par maximisation de la log-vraisemblance.
    
    Paramètres :
    - x (numpy.ndarray) : Données observées.

    Retourne :
    - numpy.ndarray : Paramètres optimisés [mu, sigma, gamma, nu] si succès.
    - None : Si l'optimisation échoue.
    """
    
    # Initialisation, contraintes et bornes des paramètres
    theta_init = [np.mean(x), np.std(x), 0, 5]
    bounds = [(None, None), (1e-6, None), (None, None), (1, None)]
    constraints = [{'type': 'ineq', 'fun': lambda theta: theta[1]}, {'type': 'ineq', 'fun': lambda theta: theta[3] - 1}]
    
    # Optimisation de la log-vraisemblance
    result = minimize(log_likelihood, theta_init, args=(x,), method='trust-constr', bounds=bounds, constraints=constraints)
    
    # paramètres optimisés 
    if result.success:
        return result.x
    else:
        print("L'optimisation a échoué.")
        return None


# Fonction de simulation

def skew_student_sim(mu, sigma, gamma, nu, size):
    """
    Simule des données suivant une loi de Skew Student.
    
    Paramètres :
    - mu : Moyenne.
    - sigma : Écart-type.
    - gamma : Paramètre de skewness.
    - nu : Degrés de liberté.
    - size : Nombre de simulations.
    
    Retourne :
    - numpy.ndarray : Valeurs simulées suivant Skew Student.
    """
    np.random.seed(3)
    # Générer T1 et T2 avec la loi de Student
    T1 = st.t.rvs(df=nu, loc=0, scale=1, size=size)
    np.random.seed(4)
    T2 = st.t.rvs(df=nu, loc=0, scale=1, size=size)
    
    # Appliquer la transformation Skew Student
    Z = mu + sigma / np.sqrt(1 + gamma**2) * (gamma * np.abs(T1) + T2)
    
    return Z

############################ Evaluation de l'ajustement ############################

from scipy.stats import gaussian_kde

def plot_skew_student_fit(log_returns, mu, sigma, gamma, nu, pdf_function):
    """
    Compare la densité empirique des log-rendements à la densité théorique 
    d'une loi Skew Student estimée.

    Parameters:
        log_returns (array-like): Série des log-rendements.
        mu (float): Moyenne estimée.
        sigma (float): Écart-type estimé.
        gamma (float): Skewness estimée.
        nu (float): Degrés de liberté estimés.
        pdf_function (callable): Fonction de densité Skew Student prenant (x, mu, sigma, gamma, nu).
    """
    x_vals = np.linspace(min(log_returns), max(log_returns), 1000)

    # Densité empirique (par KDE)
    kde = gaussian_kde(log_returns)
    empirical_density = kde(x_vals)

    # Densité théorique selon les paramètres estimés
    theoretical_density = pdf_function(x_vals, mu, sigma, gamma, nu)

    # Tracé
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, empirical_density, label='Densité empirique', color='blue')
    plt.plot(x_vals, theoretical_density, label='Densité Skew Student', color='red', linestyle='--')
    plt.title("Validation ex-ante : Densité empirique vs. Skew Student théorique")
    plt.xlabel("Log-rendements")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)
    plt.show()

########################### QQ-Plot ##################################################


def qqplot(df_observed, df_simulated):
    """
    Génère un QQ plot comparant les quantiles des données observées et simulées.
    
    Paramètres :
    - df_observed (numpy.ndarray) : Données observées.
    - df_simulated (numpy.ndarray) : Données simulées.
    
    Retourne :
    - fig (matplotlib.figure.Figure) : La figure du QQ plot générée.
    """
    quantiles_x = np.percentile(df_observed, np.linspace(0, 100, len(df_observed)))
    quantiles_y = np.percentile(df_simulated, np.linspace(0, 100, len(df_simulated)))

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(quantiles_x, quantiles_y, alpha=0.5, color='blue', label='Quantiles observés vs simulés')  # Bleu roi
    plt.plot([min(quantiles_x), max(quantiles_x)], [min(quantiles_x), max(quantiles_x)], color='red', linestyle='--', label='Ligne idéale')
    plt.title('QQ Plot : Comparaison des Quantiles Observés et Simulés')
    plt.xlabel('Quantiles Empiriques (Observés)')
    plt.ylabel('Quantiles Théoriques (Simulés)')
    plt.grid(True)
    plt.legend()
    return fig



from scipy.stats import norm

def qqplot_gaussian_skew(df_observed, mu_gauss, sigma_gauss, mu_skew, sigma_skew, gamma_skew, nu_skew):
    """
    Compare les quantiles de la loi Gaussienne et de la loi Skew Student avec les données observées.
    
    Paramètres :
    - df_observed : Données observées
    - mu_gauss, sigma_gauss : Paramètres de la loi Gaussienne
    - mu_skew, sigma_skew, gamma_skew, nu_skew : Paramètres de la loi Skew Student
    """
    quantiles_x = np.percentile(df_observed, np.linspace(0, 100, len(df_observed)))
    quantiles_y_gauss = norm.ppf(np.linspace(0, 1, len(df_observed)), mu_gauss, sigma_gauss)
    df_simulated_skew = skew_student_sim(mu_skew, sigma_skew, gamma_skew, nu_skew, len(df_observed))
    quantiles_y_skew = np.percentile(df_simulated_skew, np.linspace(0, 100, len(df_observed)))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(quantiles_x, quantiles_y_gauss, alpha=0.5, color='blue')
    ax[0].plot([min(quantiles_x), max(quantiles_x)], [min(quantiles_x), max(quantiles_x)], color='red', linestyle='--')
    ax[0].set_title('QQ Plot - Loi Gaussienne')
    ax[0].set_xlabel('Quantiles Observés')
    ax[0].set_ylabel('Quantiles Théoriques (Gaussienne)')

    ax[1].scatter(quantiles_x, quantiles_y_skew, alpha=0.5, color='green')
    ax[1].plot([min(quantiles_x), max(quantiles_x)], [min(quantiles_x), max(quantiles_x)], color='red', linestyle='--')
    ax[1].set_title('QQ Plot - Loi Skew Student')
    ax[1].set_xlabel('Quantiles Observés')
    ax[1].set_ylabel('Quantiles Théoriques (Skew Student)')

    plt.tight_layout()
    plt.show()


################ VaR Skew-Student ####################################################################

def Var_param_student(data, confidence_level=0.99):
    """Calcule le VaR paramétrique basé sur la loi de Skewed Student."""
    
    theta = optimize_parameters(data)
    if theta is None:
        print("L'optimisation a échoué. Impossible de calculer le VaR.")
        return None

    mu, sigma, gamma, nu = theta
    Z = skew_student_sim(mu, sigma, gamma, nu, len(data))
    VaR_skew = np.percentile(Z, (1 - confidence_level) * 100)
    
    print(f"La VaR Skewed Student au niveau de confiance {confidence_level*100:.1f}% est de : {VaR_skew:.4f}")
    return VaR_skew



##########################################################################################################
############################## Expected shortfall ########################################################
##########################################################################################################

def ES_Hist(returns, confidence_level=0.99):
    """
    Calcule l'Expected Shortfall (ES) historique à partir des rendements observés.
    
    Paramètres :
    - returns : Les rendements historiques des actifs.
    - confidence_level : Le niveau de confiance pour le calcul (par défaut 0.99).
    
    Retourne :
    - float : L'Expected Shortfall.
    """
    # Calcul de la VaR historique
    VaR = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Calcul de l'Expected Shortfall (ES)
    ES = returns[returns < VaR].mean()
    print(f"L'Expected Shortfall empirique historique au niveau de confiance {confidence_level*100:.1f}% est : {ES:.6f}")
    return ES


def ES_emp_gauss(train_data, confidence_level=0.99):
    """
    Calcule l'Expected Shortfall (ES) historique à partir des rendements observés.
    
    Paramètres :
    - returns : Les rendements historiques des actifs.
    - confidence_level : Le niveau de confiance pour le calcul (par défaut 0.99).
    
    Retourne :
    - float : L'Expected Shortfall.
    """
    # Calcul de la VaR gaussienne
    VaR = VaR_Gauss(train_data, alpha=0.99)
    returns = train_data['Log Return']
    
    # Calcul de l'Expected Shortfall (ES)
    ES = returns[returns < VaR].mean()
    print(f"L'Expected Shortfall empirique gaussien au niveau de confiance {confidence_level*100:.1f}% est : {ES:.6f}")
    return ES


def ES_gauss(train_data, confidence_level=0.99):
    """
    Calcule l'Expected Shortfall (ES) théorique sous hypothèse de normalité.

    Paramètres :
    - train_data : DataFrame contenant une colonne 'Log Return'
    - confidence_level : niveau de confiance (ex : 0.99)

    Retour :
    - float : ES théorique gaussien
    """
    returns = train_data['Log Return']
    mu = returns.mean()
    sigma = returns.std()
    
    z = norm.ppf(confidence_level)
    ES = mu - sigma * norm.pdf(z) / (1 - confidence_level)

    print(f"ES théorique gaussien (niveau {confidence_level*100:.1f}%) : {ES:.6f}")
    return ES


def ES_emp_skew_student(train_data, confidence_level=0.99):
    """
    Calcule l'Expected Shortfall (ES) historique à partir des rendements observés.
    
    Paramètres :
    - returns : Les rendements historiques des actifs.
    - confidence_level : Le niveau de confiance pour le calcul (par défaut 0.99).
    
    Retourne :
    - float : L'Expected Shortfall.
    """
    
    # Calcul de la VaR gaussienne
    VaR = Var_param_student(train_data['Log Return'], confidence_level=0.99)
    
    # Calcul de l'Expected Shortfall (ES)
    returns = train_data['Log Return']
    ES = returns[returns < VaR].mean()
    print(f"L'Expected Shortfall empirique skewed-student au niveau de confiance {confidence_level*100:.1f}% est : {ES:.6f}")
    return ES

def ES_skew_student(mu, sigma, gamma, nu, train_data, VaR_skew, confidence_level=0.99):
    """
    Calcule l'Expected Shortfall (ES) théorique à partir des données simulées d'une loi Skew Student.
    
    Paramètres :
    - mu : Moyenne de la loi Skew Student.
    - sigma : Écart-type de la loi Skew Student.
    - gamma : Paramètre de skewness de la loi Skew Student.
    - nu : Degrés de liberté de la loi Skew Student.
    - train_data : Données des log-rendements observés (p.ex. train_data['Log Return']).
    - VaR_skew : La VaR au niveau de confiance souhaité (p.ex. 1% ou 5%).
    - confidence_level : Niveau de confiance pour le calcul de l'ES (par défaut 99%).
    
    Retourne :
    - ES : L'Expected Shortfall empirique à partir des données simulées.
    """
    # Nombre de simulations égal au nombre d'observations dans les données d'entraînement
    T = len(train_data['Log Return'])

    # Simuler les données selon la loi Skew Student
    df_simulated = skew_student_sim(mu, sigma, gamma, nu, T)

    # Calcul de l'Expected Shortfall (ES)
    ES = df_simulated[df_simulated < VaR_skew].mean()

    print(f"L'Expected Shortfall théorique skewed-student au niveau de confiance {confidence_level*100:.1f}% est : {ES:.6f}")

    return ES


############################################################################################################
######################### Protocole de Backtesting #########################################################
############################################################################################################


#### Mise en place d'une fonction qui fait conjointement le test d'unconditional coverage et le test d'independance

def perform_backtest(data_test, var_99, significance_level_var=0.01, significance_test=0.05):
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
    data_test['VaR_exceedance'] = data_test['Log Return'] < var_99
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




def adaptive_backtesting(data_train, data_test, window_size=30, max_no_exce=252, alpha = 0.99):
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
        var_99 = VaR_Gauss(data_train, alpha=alpha)
        for i in range(window_size, len(data_test)):
            subset_test = data_test.iloc[:i]
            result_backtest, n_exceed = perform_backtest(subset_test, var_99)

            if not result_backtest:
                # Si le backtest échoue, procéder au recalibrage
                days_without_exception = 0
                jours_recalibrage.append(i)
                recalibration_count += 1
                print(f"La VaR est recalibrée à la date {data_test.index[i].strftime('%Y-%m-%d')}\n"
                f"après {i} jours.")
                print('-'*60)
                data_train = pd.concat([data_train.iloc[i:], data_test.iloc[:i]])
                result_var.append(VaR_Gauss(data_train, alpha=alpha))
                result_date_recalib.append(data_test.index[i].strftime('%Y-%m-%d'))
                data_test = data_test.iloc[i:] 
                print(f"La valeur de la nouvelle VaR recalibrée est : {VaR_Gauss(data_train, alpha=alpha)}")
                print("-"*60)
                print("\n")
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


#######################################################################################################
########################### VaR TVE  :Approche par bloc Maxima ########################################


def gumbel_plot(data , block_size=20, title=None):
    """
    Trace un Gumbel plot basé sur les maxima de blocs de taille `block_size`.

    Arguments :
    - data : array-like, série de données.
    - block_size : int, taille du bloc pour le calcul des maxima (par défaut 20).
    - title : str, titre du graphique (optionnel).

    Raises :
    - ValueError : si le nombre de blocs est inférieur à 5.
    """
    data = np.asarray(data)
    n = len(data)
    k = n // block_size
    if k < 5:
        raise ValueError("Pas assez de blocs pour un Gumbel plot fiable.")

    maxima = [max(data[i*block_size:(i+1)*block_size]) for i in range(k)]
    maxima = np.sort(maxima)

    i = np.arange(1, k + 1)
    gumbel_quantiles = -np.log(-np.log((i - 0.5) / k))

    plt.figure(figsize=(8, 6))
    plt.scatter(gumbel_quantiles, maxima, color='blue', alpha=0.8)
    plt.xlabel(r"Quantiles théoriques Gumbel")
    plt.ylabel("Maxima observés")
    plt.title(title or f"Gumbel Plot (bloc = {block_size})")
    plt.grid(True)
    plt.show()



################################### Estimation des paramètres de la GEV ############################

def block_maxima(train_data, block_size = 20):
    """
    Calcule les maxima par bloc pour une série donnée.
    
    Parameters:
    - data: Série temporelle des pertes.
    - block_size: Taille du bloc (en nombre d'observations).
    
    Returns:
    - block_max: Liste des maxima par bloc.
    """
    data = train_data["Log Return"]
    n = len(data)
    block_max = [max(data[i:i + block_size]) for i in range(0, n, block_size)]
    return np.array(block_max)


## Estimer les paramètres de la loi GEV
def fit_gev(data):
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



#####################  Validation ex-ante  de l'ajustement TVE #####################################################

def gev_plot(data, shape, loc, scale):
        """
        Valide l'ajustement de la loi GEV par QQ-plot.
        
        Parameters:
        - data: Série des maxima par bloc.
        - shape, loc, scale: Paramètres de la loi GEV.
        """
        # QQ-plot
        theoretical_quantiles = genextreme.ppf(np.linspace(0.01, 0.99, 100), shape, loc, scale)
        empirical_quantiles = np.percentile(data, np.linspace(1, 99, 100))
        
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(theoretical_quantiles, empirical_quantiles, color='blue')
        plt.plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')
        plt.xlabel('Quantiles théoriques (GEV)')
        plt.ylabel('Quantiles empiriques')
        plt.title('QQ-Plot (validation de la loi GEV ex-ante)')
        plt.grid(True)


def ks_test_gev(data, shape, loc, scale, alpha=0.05):
    """Test KS pour valider l'ajustement à une loi GEV (niveau alpha)."""
    stat, pval = kstest(data, cdf='genextreme', args=(shape, loc, scale))

    print("-" * 50)
    print("Test de Kolmogorov-Smirnov pour la GEV\n")
    print(f"Statistique KS : {stat:.4f}")
    print(f"p-value        : {pval:.4f}\n")

    if pval > alpha:
        print(f"Résultat : on ne rejette pas H₀ au seuil {alpha}.")
        print("La loi GEV est acceptable.\n")
    else:
        print(f"Résultat : on rejette H₀ au seuil {alpha}.")
        print("La loi GEV ne semble pas convenir.\n")
    print("-" * 50)


def fit_gumbel(data):
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


############################ Test de rapport de vraisemblance ######################################

from scipy import stats

def LR_test(neg_logL_gumbel, neg_logL_gev):
    """
    Test du rapport de vraisemblance entre :
    - H₀ : le modèle Gumbel est suffisant (ξ = 0)
    - H₁ : le modèle GEV est préférable (ξ ≠ 0)

    Retourne :
    - True  : si H₀ n'est pas rejetée → Gumbel préféré
    - False : si H₀ est rejetée → GEV significativement meilleur
    """
    LRT_stat = -2 * (neg_logL_gev - neg_logL_gumbel)
    p_value = 1 - stats.chi2.cdf(LRT_stat, df=1)

    print("-" * 50)
    print("Test du rapport de vraisemblance : Gumbel (H₀) vs GEV (H₁)\n")
    print(f"Statistique LRT : {LRT_stat:.4f}")
    print(f"p-value         : {p_value:.4f}\n")

    if p_value < 0.05:
        print("Résultat : le modèle GEV améliore significativement l'ajustement.")
        print("-" * 50)
        return False
    else:
        print("Résultat : pas d'amélioration significative par rapport au modèle Gumbel.")
        print("-" * 50)
        return True


########################## Calcul de la VaR Gumbel ############################################

def calcul_VaR_Gumbel ( loc, scale, block_size, alpha = 0.99):
    VaR_Gumbel = -gumbel_r.ppf(alpha**block_size, loc=loc, scale=scale)
    print("La VaR Gumbel est :", VaR_Gumbel)
    print('-'*40)
    return VaR_Gumbel



##################################################################################################
############################ Approche TVE #######################################################

def mean_excess_plot(train_data, u_min=0, u_max=None, step=0.01):
        """
        Trace le Mean Excess Plot pour déterminer un seuil u approprié.

        Parameters:
        - data: Série des pertes (rendements négatifs).
        - u_min: Seuil minimal à considérer.
        - u_max: Seuil maximal à considérer.
        - step: Pas pour l'incrémentation des seuils.

        Returns:
        - Un graphique du Mean Excess Plot.
        """
        data = train_data['Log Return']
        if u_max is None:
            u_max = np.quantile(data, 0.99)  # Ne pas considérer les valeurs trop extrêmes

        thresholds = np.arange(u_min, u_max, step)
        mean_excess = [np.mean(data[data > u] - u) for u in thresholds]

        fig = plt.figure(figsize=(10, 6))
        plt.plot(thresholds, mean_excess, 'bo-', label='Mean Excess')
        plt.axhline(0, color='red', linestyle='--', label='Zero Line')
        plt.xlabel('Seuil u')
        plt.ylabel('Moyenne des excès')
        plt.title('Mean Excess Plot')
        plt.legend()
        plt.grid()


def fit_gpd(data, u):
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


################################### Analyse ex-ante de l'ajustement GPD ####################################

def gpd_validation(data, u, shape, scale):
        """
        Validation ex-ante de l'ajustement de la GPD.

        Parameters:
        - data: Série des pertes.
        - u: Seuil choisi.
        - shape, scale: Paramètres de la GPD.

        Returns:
        - QQ-plot et PP-plot.
        """
        excess = data[data > u] - u
        n = len(excess)
        theoretical_quantiles = genpareto.ppf(np.linspace(0, 1, n), shape, loc=0, scale=scale)
        empirical_quantiles = np.sort(excess)

        # QQ-plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].scatter(theoretical_quantiles, empirical_quantiles, color='blue')
        axes[0].plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')
        axes[0].set_xlabel('Quantiles théoriques')
        axes[0].set_ylabel('Quantiles empiriques')
        axes[0].set_title('QQ-plot (validation GPD ex-ante)')
        axes[0].grid()

        # PP-plot
        theoretical_probs = genpareto.cdf(empirical_quantiles, shape, loc=0, scale=scale)
        empirical_probs = np.linspace(0, 1, n)
        axes[1].scatter(theoretical_probs, empirical_probs, color='blue')
        axes[1].plot([0, 1], [0, 1], color='red', linestyle='--')
        axes[1].set_xlabel('Probabilités théoriques')
        axes[1].set_ylabel('Probabilités empiriques')
        axes[1].set_title('PP-plot (validation GPD ex-ante)')
        axes[1].grid()

        # Affichage
        plt.tight_layout()
        return fig


from scipy.stats import genpareto, kstest
import numpy as np

def gpd_ks_test(data, u, shape, scale, alpha=0.05):
    """
    Test de Kolmogorov-Smirnov pour valider l'ajustement de la GPD aux excès.

    Paramètres :
    - data : série initiale
    - u : seuil utilisé
    - shape, scale : paramètres de la GPD ajustée
    - alpha : niveau de significativité (par défaut 5%)
    """
    excess = data[data > u] - u
    stat, pval = kstest(excess, 'genpareto', args=(shape, 0, scale))

    print("-" * 50)
    print("Test de Kolmogorov-Smirnov pour la loi GPD\n")
    print(f"Statistique KS : {stat:.4f}")
    print(f"p-value        : {pval:.4f}\n")

    if pval > alpha:
        print(f"Résultat : H₀ non rejetée au seuil {alpha}.")
        print("Conclusion : la loi GPD est acceptable pour les excès.\n")
    else:
        print(f"Résultat : H₀ rejetée au seuil {alpha}.")
        print("Conclusion : la loi GPD n'est pas satisfaisante.\n")
    print("-" * 50)



def var_tve_pot(data, u, shape, scale, alpha=0.99):
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
    return - var


######################## Protocole de calibration de u pour la méthode POT ###############################

def calibrate_u(data, alpha=0.99, step=0.0001):
        """
        Calibre automatiquement les valeurs de u.
        
        Parameters:
        - data: Log rendement.
        - alpha: Niveau de confiance.
        - u_min, u_max: plage  des valeurs de u à tester.
        - step: pas avec lequel on parcoure la plage des valeurs de u.
        
        Returns:
        - u optimal.
        """
        u_min = np.quantile(data, 0.90) 
        u_max = np.quantile(data, 0.99)  
        
        thresholds = np.arange(u_min, u_max, step)
        shapes = []
        scales = []
        var_tve_values = []

        for u in thresholds:
            excess = data[data > u] - u
            if len(excess) > 10:  # On s'assure qu'il y ait assez d'excès
                shape, loc, scale = fit_gpd(data, u)
                shapes.append(shape)
                scales.append(scale)
                var_tve_values.append(var_tve_pot(data, u, shape, scale, alpha))

        # Identification du u le plus stable
        shape_stability = np.abs(np.diff(shapes))
        scale_stability = np.abs(np.diff(scales))

        stability = shape_stability + scale_stability
        u_optimal_idx = np.argmin(stability) + 1  # Ajout de 1 pour adapter les indexes
        #plt.plot(thresholds[:-1],stability)  # on visualise à quel point les paramètres de la loi ajustée sont 
                                             #stables quand on parcoure les valeurs de u
        u_optimal = thresholds[u_optimal_idx]

        return u_optimal


#######################################################################################################
################################ VaR GARCH ###########################################################
########################################################################################################



######### Var dynamique GPD 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_and_plot_var_GPD(std_residuals, alpha, combined_fit, train_data, test_data):
    # Calibration du seuil
    u = calibrate_u(std_residuals, alpha)
    shape, loc, scale = fit_gpd(std_residuals, u)
    VaR_res = -var_tve_pot(std_residuals, u, shape, scale, alpha)

    # Paramètres du modèle
    mu, phi, omega, a, b = combined_fit.params

    # Construction de la série complète
    data = pd.concat([train_data, test_data]).copy()

    # Initialisation de mu et vol
    data["mu"] = mu + phi * data["Log Return"].shift()
    data["mu"].iloc[0] = mu
    data["vol"] = np.sqrt(omega / (1 - a - b))

    # Mise à jour dynamique de la volatilité
    for t in range(1, len(data)):
        data["vol"].iloc[t] = np.sqrt(
            omega
            + a * (data["Log Return"].iloc[t - 1] - data["mu"].iloc[t - 1]) ** 2
            + b * data["vol"].iloc[t - 1] ** 2
        )

    # Calcul de la VaR
    data["VaR"] = -1 * (data["mu"] + data["vol"] * VaR_res)

    # Tracé du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["VaR"], label="VaR", color="red", linestyle="--")
    plt.plot(data.index, data['Log Return'], label="Rendements", color="blue")

    exceedance_points = data[data["VaR"] > data['Log Return']]
    plt.scatter(exceedance_points.index, exceedance_points['Log Return'], color="red", label="Exception", zorder=5)

    plt.title("VaR vs Rendements")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()

