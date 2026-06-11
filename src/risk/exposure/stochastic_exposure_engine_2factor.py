import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

from src.curves.simulator.hull_white_2factor import HullWhite2FactorSimulator

from src.pricing.hull_white_2factor_swap_pricer import HullWhite2FactorSwapPricer

class MonteCarloExposureEngine2Factor:
    """
    MonteCarlo exposure engine using 2-factor Hull-White factor paths simulator

    HW 2-Factor Simulator module simulates factor paths x_t and y_t, and reprice swaps using bond pricing equation at each time step    
    """
    def __init__(
            self,
            pricer: HullWhite2FactorSwapPricer,
            simulator: HullWhite2FactorSimulator
    ):
        # attributes
        self.pricer = pricer
        self.simulator = simulator

    # risk analytics layer
    def simulate_pv_matrix(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ Build future values of a trade under stochastic scenario curves using 2-factor HW factor paths simulator """
        # generate simulated scenario paths
        scenario_paths = self.simulator.simulate_multi_paths(
            maturity = swap.maturity,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # HW 2-factor state variables
        x_paths = scenario_paths['Factor1']
        y_paths = scenario_paths['Factor2']
        
        # cf grid -> (n_times x 1)
        cf_grid = CashflowGridder.cf_grid(
            maturity = swap.maturity,
            steps_per_year = steps_per_year
        )

        # initialize (n_paths x n_times) array
        pv_matrix = np.zeros((n_paths, len(cf_grid)))

        for path_idx in range(n_paths):

            for time_idx, t in enumerate(cf_grid):
                
                # ensure aged swap at valuation = t
                swap_t = swap.aged_swap(valuation_time = t)

                # extract factor states
                x_t = x_paths[path_idx, time_idx]
                y_t = y_paths[path_idx, time_idx]

                # compute future values
                pv_matrix[path_idx, time_idx] = self.pricer.price(
                    swap = swap_t,
                    x_t = x_t,
                    y_t = y_t
                )
        
        return pv_matrix
    
    def expected_exposure(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4  
    ):
        """
        Return expected exposure at a future date

        Formula:
            EE_{t} = E[max(V_{t}, 0)]
        """
        # cf grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = swap.maturity,
            steps_per_year = steps_per_year
        )

        # PV matrix -> (n_paths x n_times)
        pv_matrix = self.simulate_pv_matrix(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # positive exposures
        positive_exposures = np.maximum(pv_matrix, 0)

        # expected exposures -> (1 x n_times)
        ee = positive_exposures.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'EE': ee
        })
    
    def expected_negative_exposure(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return expected negative exposure at a future date

        Formula:
            ENE_{t} = E[max(-V_{t}, 0)]
        """
        # cf grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = swap.maturity,
            steps_per_year = steps_per_year
        )

        # PV matrix -> (n_paths x n_times)
        pv_matrix = self.simulate_pv_matrix(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # negative exposures
        negative_exposures = np.where(
            pv_matrix < 0,
            pv_matrix,
            0
        )

        # expected negative exposures -> (1 x n_times)
        ene = -negative_exposures.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'ENE': ene
        })
    
    def potential_future_exposure(
            self,
            swap,
            percentile: float = 95.0,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return potential future exposure at a future date 
        
        Formula:
            PFE_{t} = Quantile_{q}(max(V_{t}, 0))
        """
        # cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = swap.maturity,
            steps_per_year = steps_per_year
        )

        # PV matrix -> (n_paths x n_times)
        pv_matrix = self.simulate_pv_matrix(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # positive exposures
        positive_exposures = np.maximum(pv_matrix, 0)

        # percentile across n_paths
        pfe = np.percentile(
            positive_exposures,
            percentile,
            axis = 0
        )

        return pd.DataFrame({
            'Times': cf_grid,
            f'PFE_{percentile:.0f}%': pfe
        })
    
    def expected_positive_exposure(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return expected positive exposure at a future date

        Formula:
            EPE = (1/N) sum_{i=1,..,n}(EE_{t_i})   
        """
        # expected exposure
        ee = self.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # expected positive exposure
        epe = ee['EE'].mean()

        return float(epe)
    
    def effective_expected_exposure(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return effective expected exposure at a future date

        EEE is the running max of EE from time t up to the time horizon 
        
        Formula:
            Effective EE(t_{i}) = max(Effective EE(t_{i-1}), EE(t_{i})) 
        """
        # expected exposure
        expected_exposure = self.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        effective_ee = []
        running_max = 0.0

        for ee in expected_exposure['EE']:

            running_max = max(running_max, ee)

            effective_ee.append(running_max)
        
        return pd.DataFrame({
            'Times': expected_exposure['Times'],
            'EffectiveEE': effective_ee
        })
    
    def effective_expected_positive_exposure(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return effective expected positive exposure at a future date
        
        Effective EPE is the weighted average of effective EE

        Formula:
            Effective EPE = (1/N) * sum_{i=1,..,n} (Effective EE(t_{i}))
        """
        # effective expected exposure
        effective_ee = self.effective_expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        return float(effective_ee['EffectiveEE'].mean())
    
    # reporting layer
    def mc_exposure_report(
            self,
            swap,
            percentile: float = 95.0,
            n_paths: int = 2500,
            steps_per_year: int = 4
    ):
        """ Summary table reporting exposure metrics under stochastic future curves """
        ee = self.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        ene = self.expected_negative_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        pfe = self.potential_future_exposure(
            swap = swap,
            percentile = percentile,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        effective_ee = self.effective_expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        report = (
            ee
            .merge(ene, on = 'Times')
            .merge(pfe, on = 'Times')
            .merge(effective_ee, on = 'Times')
        )

        return report
    
    def full_exposure_report(
            self,
            swap,
            percentile: float = 95.0,
            n_paths: int = 2500,
            steps_per_year: int = 4
    ):
        """ Full exposure report """
        report = self.mc_exposure_report(
            swap = swap,
            percentile = percentile,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        epe = self.expected_positive_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        effective_epe = self.effective_expected_positive_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        return pd.DataFrame({
            'Metric': [
                'Current Exposure',
                'Peak EE',
                'Peak ENE',
                'Peak PFE',
                'EPE',
                'Effective EPE'
            ],
            'Value': [
                report['EE'].iloc[0],
                report['EE'].max(),
                report['ENE'].max(),
                report[f'PFE_{percentile:.0f}%'].max(),
                epe,
                effective_epe
            ]
        })