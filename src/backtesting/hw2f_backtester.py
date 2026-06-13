import numpy as np
import pandas as pd

from src.curves.simulator.hull_white_2factor import HullWhite2FactorSimulator

class HW2FactorBacktester:
    """ 
    Hull-White 2-Factor model backtesting engine 
    
    Comparing historical PCA factor behavior to the model-implied simulated factor behavior

    Validation scope:
        - factor variance
        - increment variance
        - correlation
        - autocorrelation decay
    """
    def __init__(
            self,
            factor_history: pd.DataFrame,
            simulator: HullWhite2FactorSimulator,
            dt: float = 1 / 252
    ):
        # attributes
        self.factor_history = factor_history
        self.simulator = simulator
        self.dt = dt

    def historical_statistics(self):
        """ 
        Return historical factor statistics 
        
        Factor statistics in-scope:
            - Var(x)
            - Var(y)
            - Var(dx)
            - Var(dy)
            - Cov(dx, dy)
        """
        # factor series
        x = self.factor_history.iloc[:, 0]
        y = self.factor_history.iloc[:, 1]

        # daily changes
        dx = x.diff().dropna()
        dy = y.diff().dropna()

        return {
            'Var_x': float(np.var(x, ddof = 1)),
            'Var_y': float(np.var(y, ddof = 1)),
            'Var_dx': float(np.var(dx, ddof = 1)),
            'Var_dy': float(np.var(dy, ddof = 1)),
            'Cov_dxdy': float(np.corrcoef(dx, dy)[0, 1])
        }

    def modelled_statistics(
            self,
            maturity: float,
            n_paths: int = 250
    ):
        """ Return model-implied factor statistics """
        steps_per_year = int(1 / self.dt)

        # hw2f path simulations
        simulated_paths = self.simulator.simulate_multi_paths(
            maturity = maturity,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # simulated factor paths -> (n_paths x n_steps)
        factor1 = simulated_paths['Factor1']
        factor2 = simulated_paths['Factor2']

        # flatten all paths into single distribution
        x = factor1.flatten()
        y = factor2.flatten()

        # daily changes
        dx = np.diff(factor1, axis = 1).flatten()
        dy = np.diff(factor2, axis = 1).flatten()

        return {
            'Var_x': float(np.var(x, ddof = 1)),
            'Var_y': float(np.var(y, ddof = 1)),
            'Var_dx': float(np.var(dx, ddof = 1)),
            'Var_dy': float(np.var(dy, ddof = 1)),
            'Cov_dxdy': float(np.corrcoef(dx, dy)[0, 1])
        }

    def theoretical_statistics(self):
        """ 
        Return OU process theoretical variance 
        
        OU variance:
            Var(x) = (sigma1^{2}) / 2a
            Var(y) = (sigma2^{2}) / 2b
        """
        # theoretical OU variances
        var_x = (self.simulator.sigma1 ** 2) / (2 * self.simulator.a)
        var_y = (self.simulator.sigma2 ** 2) / (2 * self.simulator.b)

        return {
            'Var_x': float(var_x),
            'Var_y': float(var_y)
        }

    def variance_report(
            self,
            maturity: float,
            n_paths: int = 250
    ):
        """ Return variance report comparing historical, model-implied and theoretical factor variances """
        # factor statistics
        historical_var = self.historical_statistics()
        modelled_var = self.modelled_statistics(
            maturity = maturity,
            n_paths = n_paths
        )
        theoretical_var = self.theoretical_statistics()

        return pd.DataFrame({
            'Metric': [
                'Var_x',
                'Var_y',
                'Var_dx',
                'Var_dy'
            ],
            'Historical': [
                historical_var['Var_x'],
                historical_var['Var_y'],
                historical_var['Var_dx'],
                historical_var['Var_dy']
            ],
            'Model': [
                modelled_var['Var_x'],
                modelled_var['Var_y'],
                modelled_var['Var_dx'],
                modelled_var['Var_dy']
            ],
            'Theoretical': [
                theoretical_var['Var_x'],
                theoretical_var['Var_y'],
                np.nan,
                np.nan
            ]
        })

    def correlation_report(
            self,
            maturity: float,
            n_paths: int = 250
    ):
        """ Return correlation report comparing historical, model-implied and theoretical factor correlations """
        # factor statistics
        historical_var = self.historical_statistics()
        modelled_var = self.modelled_statistics(
            maturity = maturity,
            n_paths = n_paths
        )

        return pd.DataFrame({
            'Metric': [
                'Corr(dx, dy)'
            ],
            'Historical': [
                historical_var['Cov_dxdy']
            ],
            'Model': [
                modelled_var['Cov_dxdy']
            ],
            'Theoretical': [
                self.simulator.rho
            ]
        })
    
    def autocorrelation_profile(
            self,
            max_lag: int = 30
    ):
        """
        Comparing empirical and theoretical OU autocorrelation profiles
        
        Formula:
            Corr(x_{t}, x_{t+k}) = e^{-kappa * k * dt}

            where
                k: lag parameter
        """
        # factor series
        x = self.factor_history.iloc[:, 0].to_numpy()
        y = self.factor_history.iloc[:, 1].to_numpy()

        empirical_x = []
        empirical_y = []

        theoretical_x = []
        theoretical_y = []

        for lag in range(1, max_lag + 1):

            # empirical ACF
            acf_X = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            acf_y = np.corrcoef(y[:-lag], y[lag:])[0, 1]

            empirical_x.append(acf_X)
            empirical_y.append(acf_y)

            # theoretical OU ACF
            ou_theoretical_x = np.exp(-self.simulator.a * lag * self.dt)
            ou_theoretical_y = np.exp(-self.simulator.b * lag * self.dt)

            theoretical_x.append(ou_theoretical_x)
            theoretical_y.append(ou_theoretical_y)
        
        return pd.DataFrame({
            'Lag': [t+1 for t in range(max_lag)],
            'EmpiricalFactor1': empirical_x,
            'TheoreticalFactor1': theoretical_x,
            'EmpiricalFactor2': empirical_y,
            'TheoreticalFactor2': theoretical_y
        })
    
    def full_validation_report(
            self,
            maturity: float,
            n_paths: int = 250
    ):
        """ Return full HW 2-Factor validation report """

        return {
            'VarianceReport': self.variance_report(
                maturity = maturity,
                n_paths = n_paths
            ),
            'CorrelationReport': self.correlation_report(
                maturity = maturity,
                n_paths = n_paths
            )
        }