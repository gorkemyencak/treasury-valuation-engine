import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

class HullWhiteSimulator:
    """ Short-rate simulator """
    def __init__(
            self,
            r0: float,
            mean_reversion: float = 0.10,
            volatility: float = 0.01,
            long_run_rate: float | None = None,
            random_seed: int | None = None 
    ):
        # attributes
        self.r0 = r0
        self.a = mean_reversion
        self.sigma = volatility
        self.long_run_rate = r0 if long_run_rate is None else long_run_rate
        self.random_seed = random_seed

    def simulate_single_path(
            self,
            maturity: float,
            steps_per_year: int = 12
    ):
        """
        Hull-white short-rate single path simulation model

        Formula:
            dr_{t} = alpha * (mu - r_{t})dt + sigma*dW_{t}

            where
                - alpha: mean reversion
                - mu: long-run rate
                - sigma: volatility
                - W_{t}: brownian motion
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)


        cf_grid = CashflowGridder.cf_grid(
            maturity = maturity,
            steps_per_year = steps_per_year
        )
        
        # time step
        dt = 1 / steps_per_year

        # initialize rates
        rates = np.zeros(len(cf_grid))
        rates[0] = self.r0

        for i in range(1, len(cf_grid)):

            z = np.random.normal()

            rates[i] = (
                rates[i-1]
                + self.a * (self.long_run_rate - rates[i-1]) * dt
                + self.sigma * np.sqrt(dt) * z
            )

        return pd.DataFrame({
            'Time': cf_grid,
            'ShortRate': rates
        })
    
    def simulate_multi_paths(
            self,
            maturity: float,
            n_paths: int = 100,
            steps_per_year: int = 12
    ):
        """ Hull-white short-rate single path simulation model """
        paths = []

        for _ in range(n_paths):

            path = self.simulate_single_path(
                maturity = maturity,
                steps_per_year = steps_per_year
            )

            paths.append(path['ShortRate'].values)
        
        return np.array(paths)