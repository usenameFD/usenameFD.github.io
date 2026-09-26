import numpy as np
from scipy.optimize import broyden1
from scipy.stats import norm, gamma


def heston(stock_price, initial_vol, kappa, theta, lambd, rho, r, T, K):
    """
    Compute the option price using the Heston model.

    Parameters:
        kappa: Mean reversion speed.
        theta: Long-term variance.
        lambd: Volatility of volatility.
        T: Time to maturity.
        K: Strike price.

    Returns:
        Option price.
    """
    # Initial values
    V0 = initial_vol
    X0 = stock_price
    I = 1j  # Imaginary unit
    P, umax, N = 0, 1000, 10000
    du = umax / N

    aa = theta * kappa * T / lambd**2
    bb = -2 * theta * kappa / lambd**2

    for i in range(1, N):
        u2 = i * du
        u1 = u2 - 1j  # Complex number

        a1 = rho * lambd * u1 * I
        a2 = rho * lambd * u2 * I

        d1 = np.sqrt((a1 - kappa)**2 + lambd**2 * (u1 * I + u1**2))
        d2 = np.sqrt((a2 - kappa)**2 + lambd**2 * (u2 * I + u2**2))

        g1 = (kappa - a1 - d1) / (kappa - a1 + d1)
        g2 = (kappa - a2 - d2) / (kappa - a2 + d2)

        b1 = np.exp(u1 * I * (np.log(X0 / K) + r * T)) * ((1 - g1 * np.exp(-d1 * T)) / (1 - g1))**bb
        b2 = np.exp(u2 * I * (np.log(X0 / K) + r * T)) * ((1 - g2 * np.exp(-d2 * T)) / (1 - g2))**bb

        phi1 = b1 * np.exp(aa * (kappa - a1 - d1) + V0 * (kappa - a1 - d1) * (1 - np.exp(-d1 * T)) / (1 - g1 * np.exp(-d1 * T)) / lambd**2)
        phi2 = b2 * np.exp(aa * (kappa - a2 - d2) + V0 * (kappa - a2 - d2) * (1 - np.exp(-d2 * T)) / (1 - g2 * np.exp(-d2 * T)) / lambd**2)

        P += ((phi1 - phi2) / (u2 * I)) * du

    return K * np.real((X0 / K - np.exp(-r * T)) / 2 + P / np.pi)


def resample_particles(particles, weights):
    """
    Resample particles according to their weights.

    Parameters:
        particles: List or array of particles (state trajectories).
        weights: List or array of particle weights.

    Returns:
        resampled_particles: Resampled particles.
        resampled_indices: Indices of the resampled particles.
        resampled_weights: Uniform weights (1/M) for the resampled particles.
    """
    M = len(particles)  # Number of particles

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Compute cumulative weights
    cumulative_weights = np.cumsum(weights)

    # Generate M uniform random variables
    U = np.random.uniform(0, 1, M)

    # Resample particles
    resampled_particles = []
    resampled_indices = []
    for j in range(M):
        # Find the smallest i such that U[j] < cumulative_weights[i]
        i = np.searchsorted(cumulative_weights, U[j], side='right')
        resampled_particles.append(particles[i])
        resampled_indices.append(i)

    # Reset weights to 1/M
    resampled_weights = np.ones(M) / M

    return resampled_particles, resampled_indices, resampled_weights



def Bootstrap_Heston(C, S, K, tau, r, lamda, v_bar, rho, sigma, eta, M=100):
    
    n = len(C)  # Longueur de l'échantillon
    delta = tau/n
    
    # Etape 1: Initialisation des particules et des poids
    shape= (2*lamda*v_bar) / (eta**2)
    scale= (eta**2) / (2*lamda)
    v = np.random.gamma(shape, scale = scale, size=M) # Particules initiales
    weights = gamma.pdf(v, shape, scale)
    weights /= np.sum(weights)

    v_t = []
    for t in range(n):
        for i in range(M):
            mu = v[i] + lamda * delta * (v_bar - v[i])
            sig =  eta * np.sqrt(v[i]*delta)
            v[i] = np.abs(np.random.normal(mu, sig))
            C_t = heston(stock_price = S[t],
                         initial_vol = v[i],
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)
            weights[i] = norm.pdf(C[t], loc = C_t, scale = np.sqrt(sigma))
        weights /= np.sum(weights)
    
        resampled_particles, resampled_indices, resampled_weights = resample_particles(v, weights)
        weights = resampled_weights
        v = resampled_particles
        v_t.append(np.mean(v))
    return np.array(v_t)


def APF(C, S, K, tau, r, lamda, v_bar, rho, sigma, eta, M=100):
    
    n = len(C)  # Longueur de l'échantillon
    delta = tau/n

    # Etape 1: Initialisation des particules et des poids
    shape= (2*lamda*v_bar) / (eta**2)
    scale= (eta**2) / (2*lamda)
    x_particles = np.random.gamma(shape, scale = scale, size=M) # Particules initiales

    weights = np.ones(M) / M # Poids initiaux

    v_t = []
    for t in range(n):

        # Etape 2:
        p_mu_t = []
        for i in range(M):
            mu = x_particles[i] + lamda * delta * (v_bar - x_particles[i])
            sig =  eta * np.sqrt(x_particles[i]*delta)
            mu_t = sig * np.sqrt(2/np.pi) * np.exp(-0.5*(mu/sig)**2) + mu * (1-2*norm.cdf(-mu/sig))
        
            C_t = heston(stock_price = S[t],
                         initial_vol = mu_t,
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)
            p_mu_t.append(norm.pdf(C[t], loc = C_t, scale = sigma**0.5))
            weights[i] = weights[i] * p_mu_t[i]
            

        # Etape 3:
        resampled_particles, resampled_indices, resampled_weights = resample_particles(x_particles, weights)

        
        # Etape 4: Propagation
        for i in range(M):
            mu = resampled_particles[i] + lamda * delta * (v_bar - resampled_particles[i])
            sig =  eta * np.sqrt(resampled_particles[i]*delta)
            x_particles[i] = np.abs(np.random.normal(mu, sig))
            
            C_t_num = heston(stock_price = S[t],
                         initial_vol = x_particles[i],
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)
    
            # Etape 5 Update weights
            weights[i] = norm.pdf(C[t], C_t_num, np.sqrt(sigma)) /p_mu_t[resampled_indices[i]]
        # Normalize weights
        weights /= np.sum(weights)
        v_t.append(np.mean(x_particles))
    return np.array(v_t)

# APF Modified
def APF_modified(C, S, K, tau, r, lamda, v_bar, rho, sigma, eta, M=100):
    
    n = len(C)  # Longueur de l'échantillon
    delta = tau/n

    # Etape 1: Initialisation des particules et des poids
    shape= (2*lamda*v_bar) / (eta**2)
    scale= (eta**2) / (2*lamda)
    x_particles = np.random.gamma(shape, scale = scale, size=M) # Particules initiales

    weights = np.ones(M) / M # Poids initiaux

    v_t = []
    for t in range(n):

        # Etape 2:
        p_mu_t = []
        for i in range(M):
            mu = x_particles[i] + lamda * delta * (v_bar - x_particles[i])
            #sig =  eta * np.sqrt(x_particles[i]*delta)
            #mu_t = sig * np.sqrt(2/np.pi) * np.exp(-0.5*(mu/sig)**2) + mu * (1-2*norm.cdf(-mu/sig))
            mu_t = np.abs(mu)#
            C_t = heston(stock_price = S[t],
                         initial_vol = mu_t,
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)
            p_mu_t.append(norm.pdf(C[t], loc = C_t, scale = np.sqrt(sigma)))
            weights[i] = weights[i] * p_mu_t[i]
            

        # Etape 3:
        resampled_particles, resampled_indices, resampled_weights = resample_particles(x_particles, weights)

        
        # Etape 4: Propagation
        for i in range(M):
            mu = resampled_particles[i] + lamda * delta * (v_bar - resampled_particles[i])
            sig =  eta * np.sqrt(resampled_particles[i]*delta)
            x_particles[i] = np.abs(np.random.normal(mu, sig))
            
            C_t_num = heston(stock_price = S[t],
                         initial_vol = x_particles[i],
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)
    
            # Etape 5 Update weights
            weights[i] = norm.pdf(C[t], C_t_num, np.sqrt(sigma)) /p_mu_t[resampled_indices[i]]
        # Normalize weights
        weights /= np.sum(weights)
        v_t.append(np.mean(x_particles))
    return np.array(v_t)


def SimHeston(S0, K, tau, r, lamda, v_bar, rho, sigma, eta, size = 252):
    delta = tau/size
    S = [S0]
    v = [v_bar]
    C = [heston(stock_price = S0,
                         initial_vol = v_bar,
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K)]
    for t in range(1,size):
        Z = np.random.multivariate_normal(np.array([0,0]),
                                         np.array([[1,rho],[rho,1]]))
        mu = v[t-1] + lamda * delta * (v_bar - v[t-1])
        sig =  eta * np.sqrt(v[t-1]*delta)
        v.append(np.abs(mu + sig*Z[0]))
        S.append(S[t-1]*(1 + r * delta + np.sqrt(delta*v[t-1])*Z[1]))
        C.append(heston(stock_price = S[t],
                         initial_vol = v[t],
                         kappa = lamda,
                         theta = v_bar, 
                         lambd = eta, 
                         rho = rho, 
                         r = r, 
                         T = tau,
                         K = K) + np.sqrt(sigma)*np.random.normal(0,1))
    return S, C, v