import itertools
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from zermeloDF import ZermeloDF, ZermeloDFConfig
from zermeloPinns import ZermeloPinns, ZermeloPinnsConfig


class ZermeloNumericalTestConfig:
    def __init__(self, r: float, R: float, kappa: float, vs: float,
                 a: float, sigma_x: float, sigma_y: float,
                 X_min: float = -2.0, X_max: float = 2.0,
                 Y_min: float = -2.0, Y_max: float = 2.0, u_ex: bool = True):
        # Paramètres physiques
        self.r, self.R = r, R
        self.kappa, self.vs, self.a = kappa, vs, a
        self.sigma_x, self.sigma_y = sigma_x, sigma_y

        # Domaine
        self.X_min, self.X_max = X_min, X_max
        self.Y_min, self.Y_max = Y_min, Y_max
        self.u_ex = u_ex                      # préciser si on a une solution exacte ou non

        # Point d'évaluation de la solution
        mid = self.r + (self.R - self.r) / 2
        self.xy_val = (mid, mid)


class ZermeloNumericalTest:
    """
    Tests de convergence pour ZermeloDF.
    Évalue l'erreur sur omega_int en raffinant la grille.
    """

    def __init__(self, config: ZermeloNumericalTestConfig):
        self.config = config

        # Résultats stockés
        self.result_convergence: pd.DataFrame = None
        self.solvers: List[ZermeloDF] = []

    # Méthode utilitaire : lancer un solver pour un M donné
    def _run_solver(self, M: int) -> Tuple[ZermeloDF, float]:
        """Instancie, ajuste et chronomètre un ZermeloDF."""
        config = ZermeloDFConfig(
            M=M,
            X_min=self.config.X_min, X_max=self.config.X_max,
            Y_min=self.config.Y_min, Y_max=self.config.Y_max,
            u_ex=self.config.u_ex
        )
        solver = ZermeloDF(config)

        start = time.perf_counter()
        solver.fit(self.config.r, self.config.R, self.config.kappa, self.config.vs,
                   self.config.a, self.config.sigma_x, self.config.sigma_y)
        elapsed = time.perf_counter() - start

        return solver, elapsed

    # Test de convergence 
    def convergence_test(self, M_list: List[int]) -> pd.DataFrame:
        results = {
            'M': [], 'error': [], 'order_alpha': [], 'tcpu': []
        }
        h_list = []
        err_list = []
        self.solvers = []

        for M in M_list:
            solver, elapsed = self._run_solver(M)
            self.solvers.append(solver)
            U = solver.U
            omega_int = (solver.X**2 + solver.Y**2 > self.config.r**2) & (solver.X**2 + solver.Y**2 < self.config.R**2)
    
            if self.config.u_ex:
                U_ex = solver.u_exact(solver.X, solver.Y)
                error = (np.abs((U- U_ex)[omega_int])).max() 
            elif len(self.solvers) > 1:
                U_prec = np.zeros(U.shape)
                for i in range(M):
                    for j in range(M):
                        if omega_int[i,j]:
                            xy = (solver.X[:,0][i], solver.Y[0,:][j])
                            U_prec[i,j] = self.solvers[-2].interpolate(xy)
                error = (np.abs((U- U_prec)[omega_int])).max()
            else:
                error = 100

            h_list.append(solver.config.h)
            err_list.append(error)

            results['M'].append(M)
            results['error'].append(error)
            results['tcpu'].append(elapsed)

            k = len(err_list)
            if k > 1:
                alpha = (np.log(err_list[-2] / err_list[-1]) /
                         np.log(h_list[-2] / h_list[-1]))
            else:
                alpha = 0.0
            results['order_alpha'].append(alpha)


        self.result_convergence = pd.DataFrame(results)
        return self.result_convergence


    def run_tests(self, M_list: List[int] = None):
        """Lance convergence + stabilité et affiche les résultats."""
        if M_list is None:
            M_list = [20, 40, 80, 160]

        print("=" * 55)
        if self.config.u_ex:
            print("  TEST DE CONVERGENCE  (Cas A : solution exacte connue)")
        else:
            print("  TEST DE CONVERGENCE  (Cas B : f = 1 et solution exacte inconnue)")
        print("=" * 55)
        df_conv = self.convergence_test(M_list)
        print(df_conv)



class ZermeloPinnsTest:
    def __init__(self,
                 r: float, R: float, kappa: float, vs: float,
                 a: float, sigma_x: float, sigma_y: float,
                 M_eval: int   = 80,
                 X_min:  float = -2.0, X_max: float = 2.0,
                 Y_min:  float = -2.0, Y_max: float = 2.0):

        self.r, self.R              = r, R
        self.kappa, self.vs, self.a = kappa, vs, a
        self.sigma_x, self.sigma_y  = sigma_x, sigma_y
        self.M_eval                 = M_eval
        self.X_min, self.X_max      = X_min, X_max
        self.Y_min, self.Y_max      = Y_min, Y_max
        self.results_df: pd.DataFrame                       = None

    # Erreurs sur grille fixe
    def _eval_errors(self, solver: ZermeloPinns) -> Tuple[float, float]:
        """MSE et error de (u_pinns - u_exact) sur omega."""
        U, X, Y = solver.eval_pinns_on_grid(
            self.M_eval,
            self.X_min, self.X_max,
            self.Y_min, self.Y_max
        )
        omega = ((X**2 + Y**2 > self.r**2) &
                 (X**2 + Y**2 < self.R**2))
        U_ex  = solver.u_exact(X, Y)
        err   = (U - U_ex)[omega]
        return float(np.mean(err**2)), float(np.max(np.abs(err)))


    # Méthode principale
    def run(self,
            hidden_list: List[int],
            layers_list: List[int],
            num_iter_list:    List[int],
            n_pde_bc_list: List[Tuple[int, int]],
            delta_list: List[float],
            lr:          float = 1e-3,
            with_df:     bool  = False) -> pd.DataFrame:
            
        records      = []
   
        for k, (hidden, n_layers, delta, n_pde_bc, num_iter) in enumerate(
            zip(hidden_list, layers_list, delta_list, n_pde_bc_list, num_iter_list)
        ):
            

            # Instanciation via ZermeloPinnsConfig
            config         = ZermeloPinnsConfig(
                num_iter = num_iter,
                hidden  = hidden,
                layers  = n_layers,
                verbose = False,
                u_ex    = True,
                with_df = with_df,
                n_pde = n_pde_bc[0],
                n_bc = n_pde_bc[1]
            )
            config.lr      = lr

            # Entraînement
            solver = ZermeloPinns(config)
            start  = time.perf_counter()
            solver.fit(self.r, self.R, self.kappa, self.vs,
                       self.a, self.sigma_x, self.sigma_y)
            elapsed = time.perf_counter() - start

            # Métriques
            mse, linf = self._eval_errors(solver)
            records.append({
                'hidden':      hidden,
                'layers':      n_layers,
                'delta':       delta,
                'n_pde':       n_pde_bc[0],
                'n_bc':        n_pde_bc[1],
                'SGD iter':    num_iter,
                'MSE':         mse,
                'error':       linf,
                'tpcu': round(elapsed, 1),
            })

        self.results_df = pd.DataFrame(records)
        self._print_table()
        return self.results_df

    # Tableau
    def _print_table(self):
        print(self.results_df)
