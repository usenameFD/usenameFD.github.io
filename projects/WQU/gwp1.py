import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.datasets import fetch_california_housing
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.robust import norms

# %% [markdown]
# # Addressing Kurtosis and Heteroscedasticity
# ## Best Practices Handbook (Jupyter Notebook Implementation)

# %% [markdown]
# ## 1. Kurtosis Analysis

# %% [markdown]
# ### Definition
# Kurtosis measures the "tailedness" of a probability distribution relative to a normal distribution. The excess kurtosis is calculated as:
# 
# \[
# \text{Excess Kurtosis} = \frac{\mu_4}{\mu_2^2} - 3
# \]
# where:
# - \(\mu_4\) is the fourth central moment
# - \(\mu_2\) is the variance
# - The -3 adjustment makes normal distribution kurtosis = 0 (DeCarlo, 1997).

# %% [markdown]
# ### Description
# Positive excess kurtosis indicates heavy tails (more outliers than normal distribution), while negative suggests light tails.

# %% [markdown]
# ### Demonstration
# We'll simulate three distributions with different kurtosis properties:

np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
laplace_data = np.random.laplace(0, 1/np.sqrt(2), 1000)  # Leptokurtic
uniform_data = np.random.uniform(-np.sqrt(3), np.sqrt(3), 1000)  # Platykurtic

# Calculate kurtosis
kurtosis = {
    "Normal": stats.kurtosis(normal_data),
    "Laplace": stats.kurtosis(laplace_data),
    "Uniform": stats.kurtosis(uniform_data)
}

print("Excess Kurtosis Values:")
for dist, k in kurtosis.items():
    print(f"{dist}: {k:.2f}")

# %% [markdown]
# ### Diagram
# Visual comparison of the distributions:

plt.figure(figsize=(10, 6))
sns.kdeplot(normal_data, label=f"Normal (kurtosis={kurtosis['Normal']:.2f})")
sns.kdeplot(laplace_data, label=f"Laplace (kurtosis={kurtosis['Laplace']:.2f})", linestyle="--")
sns.kdeplot(uniform_data, label=f"Uniform (kurtosis={kurtosis['Uniform']:.2f})", linestyle=":")
plt.title("Distribution Comparison by Kurtosis")
plt.legend()
plt.show()

# %% [markdown]
# ### Diagnosis
# How to detect problematic kurtosis:

def diagnose_kurtosis(data, alpha=0.05):
    k = stats.kurtosis(data)
    stat, p = stats.kurtosistest(data)
    
    print(f"Excess Kurtosis: {k:.2f}")
    print(f"Kurtosis Test p-value: {p:.4f}")
    
    if p < alpha:
        print("Significant non-normal kurtosis detected (p < 0.05)")
    else:
        print("No significant kurtosis detected")
    
    if k > 1:
        print("Warning: Heavy tails (leptokurtic)")
    elif k < -1:
        print("Warning: Light tails (platykurtic)")

diagnose_kurtosis(normal_data)

diagnose_kurtosis(laplace_data)

# %% [markdown]
# ### Damage
# Consequences of ignoring kurtosis:
# - Inflated Type I error rates in hypothesis tests
# - Poor performance of models assuming normality
# - Underestimation of extreme event probabilities in risk models

# %% [markdown]
# ### Directions
# Solutions for handling kurtosis:

def handle_kurtosis(data):
    transformed = np.sign(data) * np.log1p(np.abs(data))
    robust_model = norms.HuberT()
    print("Suggested approaches:")
    print("1. Log transformation (for positive data)")
    print("2. Robust statistical methods (e.g., Huber loss)")
    print("3. Non-parametric tests")
    print("4. Student's t-distribution instead of normal")
    return transformed

# %% [markdown]
# ## 2. Heteroscedasticity Analysis

# %% [markdown]
# ### Definition
# Heteroscedasticity occurs when the variance of errors in a regression model is not constant across observations:
# 
# \[
# \text{Var}(\epsilon_i) \neq \text{Var}(\epsilon_j) \quad \text{for some} \quad i \neq j
# \]
# (Breusch & Pagan, 1979).

# Load data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# Simple regression
X_sm = sm.add_constant(X['MedInc'])
model = sm.OLS(y, X_sm).fit()

# %% [markdown]
# ### Diagram
# Residual plot to visualize heteroscedasticity:

plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot (Showing Heteroscedasticity)")
plt.show()

# %% [markdown]
# ### Diagnosis
# Formal tests for heteroscedasticity:

def diagnose_heteroscedasticity(model):
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan Test p-value: {bp_test[1]:.4f}")
    if bp_test[1] < 0.05:
        print("Significant heteroscedasticity detected")
    else:
        print("No significant heteroscedasticity detected")

diagnose_heteroscedasticity(model)

# %% [markdown]
# ## References
# 1. DeCarlo, L. T. (1997). "On the Meaning and Use of Kurtosis." *Psychological Methods*, **2**(3), 292–307.
# 2. Breusch, T. S., & Pagan, A. R. (1979). "A Simple Test for Heteroscedasticity and Random Coefficient Variation." *Econometrica*, **47**(5), 1287–1294.
# 3. Wooldridge, J. M. (2012). *Introductory Econometrics: A Modern Approach* (5th ed.). South-Western Cengage Learning.
