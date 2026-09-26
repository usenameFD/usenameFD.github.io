import time

import matplotlib.pyplot as plt
import numpy as np
from zermeloPinns import ZermeloPinns, ZermeloPinnsConfig


class ZermeloPinnsNumericalTest:
    def __init__(self, r=0.5, R=np.sqrt(2), kappa=0.1, a=0.2, sigma_x=0.5, sigma_y=0.2, vs=0.6):
        self.r = r
        self.R = R
        self.kappa = kappa
        self.a = a
        self.vs = vs
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        
        self.test_configs = [
            ("L2", False, True, False),
            ("Sobolev", True, True, False),
            ("Sobolev_Smooth", True, True, True),
        ]
    
    def sob_smooth_test(self, max_iter=10):
        config = ZermeloPinnsConfig(
            n_epoch=500, hidden=50, layers=3, lr=1e-3, delta=1e-3,
            n_pde=1000, n_bc=1000, verbose=False, u_ex=True, with_df=True,
            loss_sobolev=True, loss_sob_smooth=True, max_iter=max_iter
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
        labels = []
        
        for i, eta_0 in enumerate([5, 10, 20, 40]):
            for j, eta in enumerate([2, 4]):
                config.eta_0, config.eta = eta_0, 1 / eta
                model = ZermeloPinns(config)
                model.fit(self.r, self.R, self.kappa, self.vs, self.a, self.sigma_x, self.sigma_y)
                label = f'eta_0={eta_0}, eta_k={eta}^(-k)'
                color = colors[i*2 + j]
                labels.append(label)
                
                axes[0].plot(model.err, label=label, color=color, marker='o')
                axes[1].plot(model.q_factor, label=label, color=color, marker='o')
                axes[2].plot( model.tpcu, label=label, color=color, marker='o')
        
        axes[0].set(xlabel='Iteration k', ylabel='Err_k')
        axes[1].set(xlabel='Iteration k', ylabel='q_k')
        axes[2].set(xlabel='Iteration k', ylabel='Time (s)')
        
        for ax in axes[:2]:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()
                
    def smooth_vs_classical(self, vs, u_ex=True):
        models = []
        results = []
        
        for name, loss_sob, with_df, smooth in self.test_configs:
            config = ZermeloPinnsConfig(
                n_epoch=500,
                hidden=50, layers=3, lr=1e-3, delta=1e-3,
                n_pde=1000, n_bc=1000, verbose=False, u_ex=u_ex, with_df=with_df,
                loss_sobolev=loss_sob, loss_sob_smooth=smooth, max_iter=10
            )
            
            model = ZermeloPinns(config)
            start = time.time()
            model.fit(self.r, self.R, self.kappa, vs, self.a, self.sigma_x, self.sigma_y)
            elapsed = time.time() - start
            models.append(model)
            
            results.append({
                'name': name,
                'time': elapsed,
                'loss': model.loss,
                'err': model.err,
                'q_factor': getattr(model, 'q_factor', None)
            })
        
        return results, models
    
    def plot(self, results, vs):
        # Create single figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_loss, ax_err = axes
        
        colors = ['blue', 'red', 'green']
        
        for i, res in enumerate(results):
            color = colors[i % len(colors)]
            ax_loss.plot(res['loss'], label=res['name'], color=color, linewidth=1.5)
            ax_err.plot(res['err'], label=res['name'], color=color, linewidth=1.5)
        
        # Format loss subplot
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='upper right')
        
        # Format error subplot
        ax_err.set_xlabel('Iteration')
        ax_err.set_ylabel('SGD Err')
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, results):
        print("\n" + "-"*70)
        print(f"{'Method':<20} {'Time(s)':<10} {'Final MSE':<12}")
        print("-"*70)
        for res in results:
            print(f"{res['name']:<20} {res['time']:<10.2f} {res['err'][-1]:<12.4e}")