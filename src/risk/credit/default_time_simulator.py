import numpy as np
import pandas as pd

class DefaultTimeSimulator:
    """  
    DefaultTimeSimulator engine simulates default times from hazard-rate paths

    Default occurs when cumulative hazard exceeds an exponential(1) trigger

    Formula:
        tau = min{t_{i}: A_{t_{i}} >= E}

        where:
            A(t): cumulative hazard -> A(t) = ∫ λ_{s} ds
            E ~ exp(1)    
    """
    def __init__(
            self,
            random_seed: int | None = None
    ):
        # attributes
        self.random_seed = random_seed

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def simulate_default_time(
            self,
            hazard_path: np.ndarray,
            time_grid: np.ndarray
    ):
        """ 
        Simulate one default time from one hazard path 
        
        Intensity-based default simulation:
            E = -ln(U)

            where 
                U: uniform(0, 1) 
                    -> implies
                        E ~Exp(1)
        """
        if len(hazard_path) != len(time_grid):
            raise ValueError('hazard_path and time_grid must have equal length!')
        
        # time increments
        dt = np.diff(time_grid)

        # exponential trigger
        trigger = -np.log(np.random.uniform())

        # cumulative hazard
        A_t = 0.0

        for t in range(1, len(time_grid)):
            A_t += hazard_path[t] * dt[t-1] 

            if A_t >= trigger:
                tau = float(time_grid[t])
                return tau
        
        # no default before maturity
        tau = np.inf
        return tau
    
    def simulate_multi_default_times(
            self,
            hazard_paths: np.ndarray,
            time_grid: np.ndarray
    ):
        """ 
        Simulate default times across all paths 
        
        hazard_paths -> (n_paths x n_times)
        tau -> (n_paths x 1)
        """
        # initialize default times vector
        default_times = []

        for path in hazard_paths:

            tau = self.simulate_default_time(
                hazard_path = path,
                time_grid = time_grid
            )

            default_times.append(tau)
        
        return np.array(default_times, dtype = 'float')
    
    def survival_indicator_matrix(
            self,
            default_times: np.ndarray,
            time_grid: np.ndarray
    ):
        """
        Return survival indicator matrix

        I_{t, w} = 1(t < tau_{w})
        """
        # initialize indicator matrix
        s_indicator = np.zeros(
            (len(default_times), len(time_grid))
        )

        for i, tau in enumerate(default_times):

            s_indicator[i, :] = (time_grid < tau).astype('float')
        
        return s_indicator
    
    def default_indicator_matrix(
            self,
            default_times: np.ndarray,
            time_grid: np.ndarray
    ):
        """
        Return default indicator matrix

        D_{t, w} = 1(t >= tau_{w})
        """
        # initialize indicator matrix
        d_indicator = np.zeros(
            (len(default_times), len(time_grid))
        )

        for i, tau in enumerate(default_times):
            
            d_indicator[i, :] = (time_grid >= tau).astype('float')
        
        return d_indicator
    
    # reporting layer
    def summary_report(
            self,
            default_times: np.ndarray
    ):
        """ Summary statistics of simulated default times """
        # finite defaults
        finite_defaults = default_times[np.isfinite(default_times)]

        return pd.DataFrame({
            'Metric': [
                'DefaultProbability',
                'AvgDefaultTime',
                'EarliestDefault',
                'LatestDefault'
            ],
            'Value': [
                len(finite_defaults) / len(default_times),
                finite_defaults.mean() if len(finite_defaults) > 0 else np.nan,
                finite_defaults.min() if len(finite_defaults) > 0 else np.nan,
                finite_defaults.max() if len(finite_defaults) > 0 else np.nan
            ]
        })