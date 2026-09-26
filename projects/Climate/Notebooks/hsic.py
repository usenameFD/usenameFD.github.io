import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from kernel import sobolev_kernel, dirac_kernel


# Kernel Matrix
def matCal(Z, kernel):
    return kernel(Z[:, np.newaxis],Z[np.newaxis,:])

# HSIC estimator
def hsic(X,Y, kernel_x = sobolev_kernel, kernel_y = sobolev_kernel): # ¨Penser à remettre les noyaux convenants
    n = len(X)
    K, L = matCal(X, kernel_x), matCal(Y, kernel_y)
    H = np.eye(n)-1/n*np.ones(n)
    return (1/(n-1)**2)*np.trace(np.matmul(np.matmul(K,H), np.matmul(L,H)))


# Hsic test
def hsic_test(x,y, n_iter=2):
    hsic_H0 = np.array([hsic(x[:, i], y) for i in range(x.shape[1])]).squeeze()
    Hsic_info = []
    for _ in range(n_iter):
        target = np.random.permutation(y)
        mut_info = [hsic(x[:, i], target) for i in range(x.shape[1])]
        Hsic_info.append(mut_info)
    Hsic_info = np.array(Hsic_info)
    p_value = np.mean((Hsic_info>hsic_H0).astype('int'), axis=0)
    
    return hsic_H0, p_value


# Hsic test for variables in dataframe
def hsic_matrix(data):
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data)
    n = df.shape[1]
    HSIC = np.zeros((n,n))  # Initialiser la matrice HSIC avec des zéros
    p_values = np.zeros((n,n))
    
    # Calculer uniquement les éléments en dessous de la diagonale
    for i in range(n):
        hsic_h0 = hsic_test(df[:, i:], df[:, i])
        HSIC[i,i:] = hsic_h0[0]
        p_values[i,i:] = hsic_h0[1]

    # Compléter la matrice par symétrie
    HSIC += HSIC.T
    p_values += p_values.T

    return HSIC, p_values

def FeatSelect(X,Y, n_iter, alpha = 0.05, dep_test = hsic_test):
    Hsic, p_value = dep_test(X,Y)

    sns.barplot(x=list(range(np.size(p_value))), y=1-p_value)  # Use index as x-axis
    plt.ylabel("1-p-value")
    plt.xlabel("Features' index")
    plt.axhline(y=1-alpha, color='red', linestyle='--', label=f'1-α = {1-alpha}')
    plt.legend()
    plt.show()
    return Hsic, p_value, np.where(p_value < alpha) #, Hsic # Puisque sous H0 il y a indépendace
                                        # De fait on garde les features qui on une dépendance, i.e ceux qui pour lesquels on rejette H0
