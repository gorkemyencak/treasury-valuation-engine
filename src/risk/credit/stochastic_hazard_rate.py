import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

class StochasticHazardRateSimulator:
    """
    Ornstein-Uhlenbeck stochastic hazard rate simulator

    Dynamic intensity model:
        dλ_{t} = kappa (θ - λ_{t})dt + η dZ_{t}
        
        where
            kappa: mean reversion
            θ: long-run hazard rate
            η: hazard volatility
            Z_{t}: credit Brownian motion
    """
    def __init__(
            self,
            lambda0: float,
            kappa: float,
            theta: float,
            eta: float,
            random_seed: int | None = None
    ):
        # attributes
        self.lambda0 = lambda0
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_single_path(
            self,
            maturity: float,
            steps_per_year: int = 252
    ):
        """
        Simulate hazard path for a single path:
            dλ_{t} = kappa (θ - λ_{t})dt + η dZ_{t}
        """
        # cf grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = maturity,
            steps_per_year = steps_per_year
        )
        
        # increments
        dt = 1 / steps_per_year

        # time steps
        times = len(cf_grid)

        # hazard vector
        hazard = np.zeros(times)
        hazard[0] = self.lambda0

        for t in range(1, times):

            z = np.random.normal()

            hazard[t] = (
                hazard[t-1]
                + self.kappa * (self.theta - hazard[t-1]) * dt
                + self.eta * np.sqrt(dt) * z
            )

            hazard[t] = max(hazard[t], 0.0)
        
        return pd.DataFrame({
            'Times': cf_grid,
            'HazardRate': hazard
        })
    
    def simulate_multi_paths(
            self,
            maturity: float,
            n_paths: int = 250,
            steps_per_year: int = 252
    ):
        """ Multi-path hazard simulation """
        # initialize hazard paths vector
        hazard_paths = []

        for _ in range(n_paths):

            path = self.simulate_single_path(
                maturity = maturity,
                steps_per_year = steps_per_year
            )

            hazard_paths.append(path['HazardRate'].to_numpy())
        
        return np.array(hazard_paths, dtype = 'float')
    
    def survival_probability(
            self,
            hazard_path: np.ndarray,
            dt: float
    ):
        """
        Default time survival probability:
            S(t) = e^{- ∫ λ ds)
        """
        # integral term
        hazard_integral = np.cumsum(hazard_path * dt)

        # survival probability
        p_survival = np.exp(-hazard_integral)

        return p_survival
    
    def default_probability(
            self,
            hazard_path: np.ndarray,
            dt: float
    ):
        """
        Default probability:
            PD_{t} = 1 - S_{t}
        """
        # survival probability
        p_survival = self.survival_probability(
            hazard_path = hazard_path,
            dt = dt
        )

        return 1 - p_survival