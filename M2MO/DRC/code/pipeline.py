from DRC.code.dataLoader import DataLoader
from DRC.code.indexReplicator import IndexReplicator, BacktestEngine
from DRC.code.factorModel import FactorModel                
from DRC.code.utils import build_factor_correlation, stationarity_test
from DRC.code.drcModel import DRCEngine

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Pipeline:
    """Orchestre l'enchainement complet :
    data -> replication -> backtest -> preparation DRC -> branche initiale -> branche Frobenius -> comparaison.
    """

    def __init__(self, tickers, benchmark, train_period, test_period,
                 notional_total=100_000_000, K=None, M=500_000):
        self.tickers = tickers
        self.benchmark = benchmark
        self.train_period = train_period
        self.test_period = test_period
        self.notional_total = notional_total
        self.K = K
        self.M = M
        self.loader_train = None
        self.loader_test = None
        self.replicator = None
        self.backtest = None
        self.r_for_drc = None
        self.common_tickers = None
        self.PD = None
        self.JTD = None
        self.factor_model_initial = None
        self.factor_model_frobenius = None
        self.factor_comparison = None
        self.stationarity_results = None
        self.drc_engine_initial = None
        self.drc_engine_frobenius = None
        self.result_initial = None
        self.result_frobenius = None

    def step1_load_data(self):
        print('' + '=' * 60)
        print('ETAPE 1 - Chargement des donnees')
        print('=' * 60)
        self.loader_train = DataLoader(self.tickers, self.benchmark, *self.train_period).download()
        self.loader_test = DataLoader(self.tickers, self.benchmark, *self.test_period).download()

    def step2_replicate(self):
        print('' + '=' * 60)
        print('ETAPE 2 - Construction du portefeuille repliquant')
        print('=' * 60)
        r_train, r_bench_train = self.loader_train.get_returns()
        self.replicator = IndexReplicator(r_train, r_bench_train, long_only=True, w_max=0.10).fit()
        self.replicator.plot_weights()

    def step3_backtest(self):
        print('' + '=' * 60)
        print('ETAPE 3 - Backtest out-of-sample')
        print('=' * 60)
        r_test, r_bench_test = self.loader_test.get_returns()
        common = self.replicator.weights.index.intersection(r_test.columns)
        weights_aligned = self.replicator.weights.loc[common].copy()
        weights_aligned /= weights_aligned.sum()
        self.backtest = BacktestEngine(weights_aligned, r_test, r_bench_test).run()
        self.backtest.report()
        self.backtest.plot()

    def step4_prepare_drc_inputs(self, ratings_map, pd_table):
        print('' + '=' * 60)
        print('ETAPE 4 - Preparation des intrants DRC')
        print('=' * 60)
        r_test, _ = self.loader_test.get_returns()
        self.common_tickers = self.replicator.weights.index.intersection(r_test.columns)
        self.r_for_drc = r_test[self.common_tickers].copy()
        weights = self.replicator.weights.loc[self.common_tickers].copy()
        weights /= weights.sum()
        self.PD = np.array([pd_table[ratings_map.get(t, 'BBB')] for t in self.common_tickers])
        self.JTD = self.notional_total * weights.values
        print(f'Nombre de noms dans le portefeuille DRC : {len(self.common_tickers)}')
        print(f'Somme JTD : {self.JTD.sum()/1e6:.2f} M EUR')
        return self.PD, self.JTD

    def run_initial_branch(self):
        print('' + '=' * 60)
        print('BRANCHE A - Calibration initiale + DRC')
        print('=' * 60)
        self.factor_model_initial = FactorModel(self.r_for_drc, K=self.K).fit()
        self.factor_model_initial.plot_spectrum()
        self.drc_engine_initial = DRCEngine(self.factor_model_initial, self.PD, self.JTD, M=self.M)
        self.result_initial = self.drc_engine_initial.simulate(label='initial')
        return self.result_initial

    def run_frobenius_branch(self):
        print('' + '=' * 60)
        print('BRANCHE B - Calibration Frobenius + stationnarite + DRC')
        print('=' * 60)
        self.factor_model_frobenius = FactorModelFrobenius(self.r_for_drc, K=self.K).fit()
        tester = FactorStationarityTester(self.factor_model_frobenius.factor_scores)
        self.stationarity_results = tester.run()
        print('Tests de stationnarite des facteurs ACP :')
        print(self.stationarity_results)
        self.drc_engine_frobenius = DRCEngine(self.factor_model_frobenius, self.PD, self.JTD, M=self.M)
        self.result_frobenius = self.drc_engine_frobenius.simulate(label='frobenius')
        return self.result_frobenius

    def compare_approaches(self):
        print('' + '=' * 60)
        print('ETAPE 5 - Comparaison des deux approches')
        print('=' * 60)
        self.factor_comparison = FactorModelComparator(self.factor_model_initial, self.factor_model_frobenius)
        print(self.factor_comparison.summary())
        self.factor_comparison.plot_correlation_comparison()
        drc_comparison = pd.DataFrame({
            'Approche': ['Initiale', 'Frobenius'],
            'DRC 99.9% (M EUR)': [self.result_initial.DRC_999 / 1e6, self.result_frobenius.DRC_999 / 1e6],
            'EL (M EUR)': [self.result_initial.EL / 1e6, self.result_frobenius.EL / 1e6],
            'VaR 99% (M EUR)': [self.result_initial.VaR_99 / 1e6, self.result_frobenius.VaR_99 / 1e6],
            'VaR 99.99% (M EUR)': [self.result_initial.VaR_9999 / 1e6, self.result_frobenius.VaR_9999 / 1e6]
        })
        print('Comparaison DRC :')
        print(drc_comparison)
        return drc_comparison

    def run_full(self, ratings_map, pd_table):
        self.step1_load_data()
        self.step2_replicate()
        self.step3_backtest()
        self.step4_prepare_drc_inputs(ratings_map, pd_table)
        self.run_initial_branch()
        self.run_frobenius_branch()
        self.compare_approaches()
        return self.PD, self.JTD
