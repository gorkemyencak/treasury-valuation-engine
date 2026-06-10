import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

class HullWhite2FactorSimulator:
    """ 
    Gaussian 2-factor Hull-White curve simulator 
    
    HullWhite2FactorSimulator generates factor paths that will then be used to construct the full yield curve
    """
    def __init__(
            self,
            r0: float,
            a: float = 0.10,
            b: float = 0.50,
            sigma1: float = 0.010,
            sigma2: float = 0.005,
            rho: float = 0.25,
            random_seed: int | None = None 
    ):
        # attributes
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(self.random_seed)

    def simulate_single_path(
            self,
            maturity: float,
            steps_per_year: int = 12
    ):
        """
        Hull-White 2-factor single path simulation model

        Simulate:
            - x_{t}
            - y_{t}
            - r_{t}

        Two-factor State Process Formula:
            r_{t} = r_{0} + x_{t} + y_{t}

            dx = -a * x * dt + sigma_{1} * dW_{1}

            dy = -b * y * dt + sigma_{2} * dW_{2}  

            corr(dW_{1}, dW_{2}) = rho

            where
                - x: long-term/level factor
                - y: short-term/slope factor
                - a: slow mean reversion
                - b: fast mean reversion
                - sigma_{1}: long-end volatility
                - sigma_{2}: short-end volatility
                - rho: emprical correlation
        """
        # cashflow grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = maturity,
            steps_per_year = steps_per_year
        )

        # time step
        dt = 1 / steps_per_year

        # grid length
        n_steps = len(cf_grid)

        # initialize factors
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        
        # initialize rates
        rates = np.zeros(n_steps)
        rates[0] = self.r0

        for i in range(1, n_steps):

            z1 = np.random.normal()
            z2 = np.random.normal()

            # correlated Brownian shocks
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            # Euler discretization
            x[i] = (
                x[i-1]
                - self.a * x[i-1] * dt
                + self.sigma1 * np.sqrt(dt) * w1
            )

            y[i] = (
                y[i-1]
                - self.b * y[i-1] * dt
                + self.sigma2 * np.sqrt(dt) * w2
            )

            rates[i] = self.r0 + x[i] + y[i]
        
        return pd.DataFrame({
            'Time': cf_grid,
            'Factor1': x,
            'Factor2': y,
            'ShortRate': rates
        })
    
    def simulate_multi_paths(
            self,
            maturity: float,
            n_paths: int = 250,
            steps_per_year: int = 12
    ):
        """
        Hull-White 2-factor n_path simulation model

        Returns:
            - factor1_paths
            - factor2_paths
            - rate_paths
        """
        # initialize paths -> (n_paths, n_steps)
        factor1_paths = []
        factor2_paths = []
        rate_paths = []

        for _ in range(n_paths):

            path = self.simulate_single_path(
                maturity = maturity,
                steps_per_year = steps_per_year
            )

            factor1_paths.append(path['Factor1'].values)
            factor2_paths.append(path['Factor2'].values)
            rate_paths.append(path['ShortRate'].values)

        return {
            'Factor1': np.array(factor1_paths),
            'Factor2': np.array(factor2_paths),
            'ShortRate': np.array(rate_paths)
        }