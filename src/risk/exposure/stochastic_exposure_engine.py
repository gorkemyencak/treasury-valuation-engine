import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

from src.pricing.swap_pricer import SwapPricer

from src.curves.zero_curve import ZeroCurve
from src.curves.projection_curve import ProjectionCurve
from src.curves.shocks.curve_shocks import CurveShockEngine
from src.curves.simulator.hull_white import HullWhiteSimulator

class MonteCarloExposureEngine:
    """ 
    Projecting future exposures to the counterparty over the life of a trade 
    
        -> Stochastic future curves

    Compute:
        * Exposure
        * Expected Exposure (EE)
        * Expected Negative Exposure (ENE)
        * Potential Future Exposure (PFE)
        * Expected Positive Exposure (EPE)    
    """
    def __init__(
            self,
            pricer: SwapPricer,
            simulator: HullWhiteSimulator,
            zero_curve: ZeroCurve
    ):
        # attributes
        self.pricer = pricer
        self.simulator = simulator
        self.zero_curve = zero_curve

    # helper functions
    def _scenario_curve(
            self,
            simulated_rate
    ):
        """ Build a scenario curve by stochastic parallel shock """
        r0 = min(self.zero_curve.zero_rates.values())

        # stochastic shock
        shift_bps = int((simulated_rate - r0) * 10000)

        return CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = shift_bps
        )
    
    def _build_shocked_curves(
        self,
        shocked_zero_curve: ZeroCurve
    ):
        """ Build shocked discount and projection curves from shocked zero curve """
        # discount curve
        shocked_dc = shocked_zero_curve.to_discount_curve()

        # projection curve
        shocked_proj = ProjectionCurve(discount_curve = shocked_dc)

        return shocked_dc, shocked_proj
    
    def _scenario_pv(
            self,
            swap,
            simulated_rate
    ):
        """ Repricing swap under scenario curve """
        # construct zero curve under scenario
        scenario_zero_curve = self._scenario_curve(simulated_rate = simulated_rate)

        # build discount & projection curves
        scenario_dc, scenario_proj = self._build_shocked_curves(shocked_zero_curve = scenario_zero_curve)

        # build pricer engine
        scenario_pricer = SwapPricer(
            discount_curve = scenario_dc,
            projection_curve = scenario_proj
        )

        return scenario_pricer.price(swap = swap)
    
    # risk analytics layer
    def simulate_pv_matrix(
            self,
            swap,
            n_paths = 2500,
            steps_per_year = 4
    ):
        """ Build future values of a trade under stochastic scenario curves """
        # generate simulated scenario paths -> (n_paths x n_times)
        scenario_paths = self.simulator.simulate_multi_paths(
            maturity = swap.maturity,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # initialize (n_paths x n_times) array
        pv_matrix = np.zeros_like(scenario_paths)

        for path in range(scenario_paths.shape[0]):
            
            for time in range(scenario_paths.shape[1]):

                pv_matrix[path, time] = self._scenario_pv(
                    swap = swap,
                    simulated_rate = scenario_paths[path, time]
                )
        
        return pv_matrix

    def expected_exposure(
            self,
            swap,
            n_paths = 2500,
            steps_per_year = 4
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
        expected_exposures = positive_exposures.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'EE': expected_exposures
        })
    
    def expected_negative_exposure(
            self,
            swap,
            n_paths = 2500,
            steps_per_year = 4
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
        negative_exposures = np.minimum(pv_matrix, 0)

        # expected negatve exposures -> (1 x n_times)
        expected_negative_exposures = -1 * negative_exposures.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'ENE': expected_negative_exposures
        })
    
    def potential_future_exposure(self):
        return None

    def expected_positive_exposure(self):
        return None









