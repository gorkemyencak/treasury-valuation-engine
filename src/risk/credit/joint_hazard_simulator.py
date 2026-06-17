import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

class JointHazardSimulator:
    """
    Jointly simulatig market risk factors and hazard intensity

    JointHazardSimulator class builds 3-factor Gaussian system:
        X_{t} = [x_{t}, y_{t}, λ_{t}]

    Simulating:
        - x_{t}
        - y_{t}
        - λ_{t}
        - r_{t}

    Market risk:
        d_x{t} = - a x_{t} dt + sigma1 dW_{1}
        d_y{t} = - b y_{t} dt + sigma2 dW_{t}
    
    Credit risk:
        dλ_{t} = kappa (θ - λ_{t})dt + η dZ_{t}

    Short-rate:
        r_{t} = r_{0} + x_{t} + y_{t}

    Correlations:
        Corr(dW_{1}, dW_{2}) = rho_{xy}
        Corr(dW_{1}, dZ) = rho_{xλ}
        Corr(dW_{2}, dZ) = rho_{yλ}

        -> Σ = [[1          rho_{xy}   rho_{xλ}]
                [rho_{xy}   1          rho_{yλ}]
                [rho_{xλ}   rho_{yλ}          1]]

    Cholesky decomposition:
        LL^{T} = Σ
            -> epsilon = Lz 
               
                where
                z ~ N(0, 1)
    """
    def __init__(
            self,
            r0: float,
            lambda0: float,
            a: float,
            b: float,
            sigma1: float,
            sigma2: float,
            kappa: float,
            theta: float,
            eta: float,
            rho_xy: float,
            rho_x_lambda: float,
            rho_y_lambda: float,
            random_seed: int | None = None
    ):
        # attributes
        self.r0 = r0
        self.lambda0 = lambda0
        
        # hw2f attributes
        self.a = a
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # hazard attributes
        self.kappa = kappa
        self.theta = theta
        self.eta = eta

        # correlation attributes
        self.rho_xy = rho_xy
        self.rho_x_lambda = rho_x_lambda
        self.rho_y_lambda = rho_y_lambda

        if random_seed is not None:
            np.random.seed(random_seed)

        self.corr_matrix = np.array([
            [         1.0,       rho_xy, rho_x_lambda],
            [      rho_xy,          1.0, rho_y_lambda],
            [rho_x_lambda, rho_y_lambda,          1.0]
        ])

        # cholesky decomposition 
        self.cholesky = np.linalg.cholesky(self.corr_matrix)

    def simulate_single_path(
            self,
            maturity: float,
            steps_per_year: int = 252
    ):
        """ Simulate joint hazard simulator for a single path """
        # cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = maturity,
            steps_per_year = steps_per_year
        )

        # increments
        dt = 1.0 / steps_per_year

        # time steps
        times = len(cf_grid)

        # vectors
        x = np.zeros(times)
        y = np.zeros(times)
        short_rates = np.zeros(times)
        hazard = np.zeros(times)

        short_rates[0] = self.r0
        hazard[0] = self.lambda0

        for t in range(1, times):

            z = np.random.normal(size = 3)

            shocks = self.cholesky @ z

            w1 = shocks[0]
            w2 = shocks[1]
            wz = shocks[2]

            # HW 2-factor
            x[t] = (
                x[t-1]
                - self.a * x[t-1] * dt
                + self.sigma1 * np.sqrt(dt) * w1
            )

            y[t] = (
                y[t-1]
                - self.a * y[t-1] * dt
                + self.sigma2 * np.sqrt(dt) * w2
            )

            # short-rate
            short_rates[t] = self.r0 + x[t] + y[t]

            # hazard rate
            hazard[t] = (
                hazard[t-1]
                + self.kappa * (self.theta - hazard[t-1]) * dt
                + self.eta * np.sqrt(dt) * wz
            )

            hazard[t] = max(hazard[t], 0.0)
        
        return pd.DataFrame({
            'Times': cf_grid,
            'Factor1': x,
            'Factor2': y,
            'ShortRate': short_rates,
            'HazardRate': hazard
        })

    def simulate_multi_paths(
            self,
            maturity: float,
            n_paths: int = 250,
            steps_per_year: int = 252
    ):
        """ Multi-path joint hazard simulation """
        # initialize vectors
        factor1_paths = []
        factor2_paths = []
        shortrate_paths = []
        hazard_paths = []

        for _ in range(n_paths):

            path = self.simulate_single_path(
                maturity = maturity,
                steps_per_year = steps_per_year
            )

            # HW 2-factor paths
            factor1_paths.append(path['Factor1'].to_numpy())
            factor2_paths.append(path['Factor2'].to_numpy())

            # Short-rate paths
            shortrate_paths.append(path['ShortRate'].to_numpy())

            # Hazard paths
            hazard_paths.append(path['HazardRate'].to_numpy())
        
        return {
            'Factor1': np.array(factor1_paths, dtype = 'float'),
            'Factor2': np.array(factor2_paths, dtype = 'float'),
            'ShortRate': np.array(shortrate_paths, dtype = 'float'),
            'HazardRate': np.array(hazard_paths, dtype = 'float')
        }