import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from scipy.optimize import minimize


# VaR student
# Densité
def f_skew_student(x, mu, sigma, gamma, nu):
    """
    Fonction de densité de la loi Skew-Student.

    Paramètres :
    - x : nombre réel
    - mu : moyenne
    - sigma : échelle
    - gamma : asymétrie
    - nu : degrés de liberté

    Retour :
    - Quantiles de la Skew-Student
    """
    arg = (x-mu)/sigma
    arg2 = gamma*arg*np.sqrt((nu+1)/(arg**2 + nu))
    f = st.t.pdf(x, df = nu, loc = mu, scale = sigma)
    F = st.t.cdf(arg2, df = nu + 1)
    return 2*f*F

def skew_student_ppf(u, mu, sigma, gamma, nu):
    """
    Fonction quantile inverse (PPF) de la loi Skew-Student.

    Paramètres :
    - u : valeurs uniformes [0,1] (probabilités)
    - mu : moyenne
    - sigma : échelle
    - gamma : asymétrie
    - nu : degrés de liberté

    Retour :
    - Quantiles de la Skew-Student
    """
    q_t = st.t.ppf(u, df=nu)  # Quantiles de Student-t
    skew_q = mu + sigma * q_t * (1 + gamma * np.sign(q_t))  # Ajustement de l'asymétrie
    return skew_q

def skew_student_cdf(x, mu, sigma, gamma, nu):
    """
    Fonction de répartition cumulative (CDF) pour la loi Skew-Student.

    Paramètres :
    - x : valeur où évaluer la CDF
    - mu : moyenne (location)
    - sigma : échelle (scale)
    - gamma : asymétrie (skewness)
    - nu : degrés de liberté (tail heaviness)

    Retour :
    - Probabilité cumulative F(x)
    """
    S = (x - mu) / sigma  # Transformation standardisée

    # Si S est un tableau, appliquez le calcul élément par élément
    if isinstance(S, np.ndarray):  # S est un tableau numpy
        result = np.where(S < 0, 2 * st.t.cdf(S / (1 + gamma), df=nu), 1 - 2 * (1 - st.t.cdf(S / (1 - gamma), df=nu)))
    else:  # S est un scalaire
        if S < 0:
            result = 2 * st.t.cdf(S / (1 + gamma), df=nu)
        else:
            result = 1 - 2 * (1 - st.t.cdf(S / (1 - gamma), df=nu))
    
    return result

def log_likelihood(theta, x):
    mu, sigma, gamma, nu = theta
    pdf_values = f_skew_student(x, mu, sigma, gamma, nu)
    log_lik = np.sum(np.log(pdf_values))
    return -log_lik

def optimize_parameters(x):
    """Optimize parameters using 'trust-constr' method"""
    # Initial guess for [mu, sigma, gamma, nu]
    theta_init = [np.mean(x), np.std(x), 0, 5]  # [mu, sigma, gamma, nu]
    
    # Bounds: sigma > 0 and nu > 1
    bounds = [(None, None), (1e-6, None), (None, None), (1, None)]  # bounds for each parameter
    
    # Constraints to ensure parameters remain valid (for sigma > 0 and nu > 1)
    constraints = [{'type': 'ineq', 'fun': lambda theta: theta[1]},  # sigma > 0
                   {'type': 'ineq', 'fun': lambda theta: theta[3] - 1}]  # nu > 1
    
    # Use the 'trust-constr' method for optimization
    result = minimize(log_likelihood, theta_init, args=(x,), method='trust-constr', bounds=bounds, constraints=constraints)
    
    # Check the optimization result
    if result.success:
        return result.x  # Return the optimized parameters
    else:
        print("Optimization failed.")
        return None
    
def skew_student_sim(mu, sigma, gamma, nu, size):
    T1 = st.t.rvs(df=nu, loc=0, scale=1, size=size)
    T2 = st.t.rvs(df=nu, loc=0, scale=1, size=size)
    Z = mu + sigma/np.sqrt(1+gamma**2) * (gamma*np.abs(T1)+T2)
    return Z