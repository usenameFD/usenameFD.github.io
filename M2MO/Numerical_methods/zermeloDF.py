from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


class ZermeloDFConfig:
    def __init__(self, M: int, X_min: float, X_max: float, Y_min: float,
                       Y_max: float, max_iter: float = 100, tol: float = 1e-3,
                       u_ex: bool = True):
        self.M = M
        self.X_min = X_min
        self.X_max = X_max
        self.Y_min = Y_min
        self.Y_max = Y_max
        self.h = (X_max - X_min) / (M + 1)
        self.max_iter = max_iter
        self.tol = tol
        self.u_ex = u_ex

class ZermeloDF:
    def __init__(self, config: ZermeloDFConfig):
        self.r = None
        self.R = None
        self.kappa = None
        self.vs = None
        self.a = None
        self.sigma_x = None
        self.sigma_y = None
        self.config = config
        self.U, self.X, self.Y  = None, None, None

    def vc_func(self, x: float, y: float)->float:
        return (1.0 - self.a*np.sin(np.pi*(x**2 + y**2 - self.r**2)
                /(self.R**2 - self.r**2)))

    def f_func(self, x: float, y: float)->float:
        if self.config.u_ex:
            k = 0.5 * np.pi/(self.R**2 - self.r**2)
            arg = k*(x**2 + y**2 - self.r**2)
            u_x = 2*k * x * np.cos(arg)
            u_y = 2*k * y * np.cos(arg)
            u_xx = - 4*k * x**2 * np.sin(arg) + 2*k * np.cos(arg)
            u_yy = - 4*k * y**2 * np.sin(arg) + 2*k * np.cos(arg)
            grad_norm_1 = np.abs(u_x) + np.abs(u_y)
            grad_norm_2 = np.sqrt(u_x**2 + u_y**2)
            return (-0.5*(self.sigma_x**2 * u_xx + self.sigma_y**2 * u_yy) 
                    - self.vc_func(x, y)*u_x + self.vs*grad_norm_2
                    - self.kappa*grad_norm_1)
        else:
            return 1.0
    
    def u_exact(self, x:float, y: float)->float:
        return np.sin(np.pi/2 * (x**2 + y**2 - self.r**2)
                /(self.R**2 - self.r**2))

    def _maillage(self):
        # Maillage
        x = self.config.X_min + self.config.h * np.arange(1, self.config.M + 1)
        y = self.config.Y_min + self.config.h * np.arange(1, self.config.M + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        r_node = np.sqrt(X**2 + Y**2) # Distance au centre de chaque point de la grille

        # Régions du maillage
        gamma_r = (r_node <= self.r)        # disque intérieur
        gamma_R = (r_node >=self.R)        # disque extérieur

        # vc et f sur la grille
        V = np.zeros((self.config.M, self.config.M))
        F = np.zeros((self.config.M, self.config.M))
        for i in range(self.config.M):
            for j in range(self.config.M):
                V[i, j] = self.vc_func(x[i], y[j])
                F[i, j] = self.f_func(x[i], y[j])

        # discrétisation du gradient
        D = np.diag(np.ones(self.config.M-1), 1) - np.diag( np.ones(self.config.M-1), -1)
        D[0, :] = 0  # Conditions aux bords
        D[-1, :] = 0
        D = D / (2*self.config.h)
        return X, Y, gamma_r, gamma_R, V, F, D
    
    def _idx(self, i, j):
        return i * self.config.M + j

    def newton_semi_smooth(self)->Tuple[np.array, np.array, np.array]:
        X, Y, gamma_r, gamma_R, V, F, D = self._maillage()
        U_prec = np.zeros((self.config.M, self.config.M))
        U_prec[gamma_R] = 1.0  # Conditions aux bords
        U_prec[gamma_r] = 0.0

        for _ in range(self.config.max_iter):
            # gradient de U
            Ux = D @ U_prec              
            Uy = U_prec @ D.T            
            grad_norm = np.sqrt(Ux**2 + Uy**2)
            
            # Alpha (gradient normalisé)
            eps = 1e-12
            alpha_x = np.where(grad_norm > eps, Ux / grad_norm, 0.0)
            alpha_y = np.where(grad_norm > eps, Uy / grad_norm, 0.0)
            
            # Beta (signe du gradient)
            beta_x = np.sign(Ux)
            beta_y = np.sign(Uy)
            
            # Ecriture du système linéaire A @ U_curr = b
            # Taille de la matrice A: N = M*M
            N = self.config.M**2
            A_matrix = lil_matrix((N, N))
            b = np.zeros(N)

            for i in range(self.config.M):
                for j in range(self.config.M):
                    I = self._idx(i, j)

                    if gamma_r[i, j] or gamma_R[i, j]:
                        # Conditions de bord (Dirichlet)
                        A_matrix[I, I] = 1.0
                        if gamma_r[i, j]:
                            b[I] = 0.0
                        else:
                            b[I] = 1.0
                    else:
                        # Intérieur du dommaine Omega
                        # Termes en x
                        if i > 0 and i < self.config.M-1:
                            A_matrix[I, self._idx(i+1, j)] += -0.5 * self.sigma_x**2 * (1/self.config.h**2) + (1/(2*self.config.h)) * ( -V[i, j] + self.vs * alpha_x[i, j] - self.kappa * beta_x[i, j])
                            A_matrix[I, self._idx(i, j)]   += self.sigma_x**2 * (1/self.config.h**2)
                            A_matrix[I, self._idx(i-1, j)] += -0.5 * self.sigma_x**2 * (1/self.config.h**2) - (1/(2*self.config.h)) * ( -V[i, j] + self.vs * alpha_x[i, j] - self.kappa * beta_x[i, j])    
                        
                        # Termes en y
                        if j > 0 and j < self.config.M-1:
                            A_matrix[I, self._idx(i, j+1)] += -0.5 * self.sigma_y**2 * (1/self.config.h**2) +  (1/(2*self.config.h)) * (self.vs * alpha_y[i, j] - self.kappa * beta_y[i, j])
                            A_matrix[I, self._idx(i, j)]   += self.sigma_y**2 * (1/self.config.h**2)
                            A_matrix[I, self._idx(i, j-1)] += -0.5 * self.sigma_y**2 * (1/self.config.h**2) - (1/(2*self.config.h)) * ( self.vs * alpha_y[i, j] - self.kappa * beta_y[i, j] )
                        
                        # f(xi, yj)
                        b[I] = F[i, j]
            
            # Convertion auformat CSR format pour résoudre le système de façon optimale
            A_csr = A_matrix.tocsr()
            
            # Solve linear system
            u_new_flat = spsolve(A_csr, b)
            U_new = u_new_flat.reshape(self.config.M, self.config.M)
            
            # Enforce boundary conditions
            U_new[gamma_r] = 0.0
            U_new[gamma_R] = 1.0
            
            # Check convergence
            diff = np.linalg.norm(U_new - U_prec) 
            
            if diff < self.config.tol:
                U_prec = U_new.copy()
                break
            U_prec = U_new.copy()
        return U_prec, X, Y # solution finale
    
    def fit(self, r: float, R: float, kappa: float, vs: float,
                       a: float, sigma_x: float, sigma_y: float):
        self.r = r
        self.R = R
        self.kappa = kappa
        self.vs = vs
        self.a = a
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.U, self.X, self.Y = self.newton_semi_smooth()

    def interpolate(self, xy):
        i, j = 0, 0
        while xy[0] > self.X[:,0][i+1]:
            i += 1
        while xy[1] > self.Y[0,:][j+1]:
            j += 1
        # Interpolation entre s[i] et s[i+1]
        s = (xy[0] - self.X[:,0][i+1]) / self.config.h
        t = (xy[1] - self.Y[0,:][j+1]) / self.config.h
        U_val = s*t * self.U[i, j] + s*(1-t) * self.U[i, j+1] + (1-s)*t * self.U[i+1, j] + (1-s)*(1-t) * self.U[i+1, j+1]
        return U_val

    def plot_solution(self):
        omega = (self.X**2 + self.Y**2 >= self.r**2) & (self.X**2 + self.Y**2 <= self.R**2)
        X, Y, U = self.X, self.Y, self.U
        U[~omega] = np.nan

        # Plot results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, U, levels=20, cmap='plasma')
        plt.colorbar(label='u(x,y)')
        plt.title('Solution u(x,y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')

        # Plot a cross-section
        plt.subplot(1, 2, 2)
        # Take a slice along y=0 (approximately)
        j_mid = self.config.M // 2
        plt.plot(X[:,0], U[:, j_mid], 'b-', label='Numerical solution')

        if self.config.u_ex:
            U_exact = self.u_exact(self.X, self.Y)
            U_exact[~omega] = np.nan
            plt.plot(X[:,0], U_exact[:, j_mid], 'r-', label='Exact solution')

        plt.xlabel('x')
        plt.ylabel('u(x,0)')
        plt.title('Cross-section at y=0')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(16, 6))
    
        # Solution numérique
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, U, cmap='plasma', edgecolor='none', alpha=0.7)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        ax1.set_title('Solution numérique')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        
        # Solution exacte
        if self.config.u_ex:
            ax2 = fig.add_subplot(132, projection='3d')
            surf2 = ax2.plot_surface(X, Y, U_exact, cmap='plasma', edgecolor='none')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('u(x,y)')
            ax2.set_title('Solution exacte')
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        
            # Erreur
            ax3 = fig.add_subplot(133, projection='3d')
            error = np.abs(U - U_exact)
            error[~omega] = np.nan
            surf3 = ax3.plot_surface(X, Y, error, cmap='hot', edgecolor='none')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('Erreur')
            ax3.set_title('Erreur |num - exact|')
            fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
        
        plt.tight_layout()
        plt.show()