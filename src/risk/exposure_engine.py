import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

from src.pricing.swap_pricer import SwapPricer

class ExposureEngine:
    """ 
    Projecting future exposures to the counterparty over the life of a trade 
    
        -> Deterministic curve assumption

    Compute:
        * Exposure
        * EE(t)
    
        for interest rate swaps
    """
    def __init__(
            self,
            pricer
    ):
        # attributes
        self.pricer = pricer

    # helper functions
    def _remaining_payment_dates(
            self,
            payment_dates,
            valuation_time
    ):
        """ Return cashflow grid of an instrument disregarding the cashflows before valuation time """
        return [
            date
            for date in payment_dates
            if date > valuation_time
        ]
    
    def _fixed_leg_pv(
            self,
            swap,
            fixed_dates
    ):
        """
        Return fixed-leg PV of a swap

        Fixed-leg PV:
            PV_{fixed} = N * K * sum_{i=1,..,n} (alpha_{i} * DF(T_{i}))

            where
                - N: notional amount
                - K: fixed swap rate
                - alpha_{i}: accrual fraction
                - DF(T_{i}): discount factor
        """
        pv_fixed = 0.0

        accrual = 1.0 / self.pricer.freq_map[swap.fixed_freq]

        fixed_rate = swap.fixed_rate / 100.0

        for t in fixed_dates:

            df = self.pricer.discount_curve.get_discount_factor(maturity = t)

            pv_fixed += accrual * df
        
        pv_fixed *= swap.notional * fixed_rate

        return pv_fixed
    
    def _float_leg_pv(
            self,
            swap,
            float_dates,
            valuation_time
    ):
        """
        Return float-leg PV of a swap

        Float-leg PV:
            PV_{float} = N * sum_{i=1,..,n} (f_{i} * alpha_{i} * DF(T_{i}))

            where
                - N: notional amount
                - f_{i}: forward rate (projection curve)
                - alpha_{i}: accrual fraction
                - DF(T_{i}): discount factor
        """
        pv_float = 0.0

        accrual = 1.0 / self.pricer.freq_map[swap.float_freq]

        # start time (= valuation time)
        t1 = valuation_time

        for t in float_dates:

            # end time
            t2 = t

            # compute forward rate
            forward_rate = self.pricer.projection_curve.get_forward_rate(
                start = t1,
                end = t2
            )

            df = self.pricer.discount_curve.get_discount_factor(maturity = t2)

            pv_float += forward_rate * accrual * df

            t1 = t2
        
        pv_float *= swap.notional

        return pv_float

    def _forward_pv(
            self,
            swap,
            valuation_time
    ):
        """ Return mark-to-market value of a trade at a future valuation time """
        # fixed & floating-leg remaining payment dates
        fixed_leg_dates = self._remaining_payment_dates(
            payment_dates = swap.fixed_schedule.generate_schedule(),
            valuation_time = valuation_time
        )

        floating_leg_dates = self._remaining_payment_dates(
            payment_dates = swap.float_schedule.generate_schedule(),
            valuation_time = valuation_time
        )

        # fixed-leg PV
        pv_fixed = self._fixed_leg_pv(
            swap = swap,
            fixed_dates = fixed_leg_dates
        )

        # float-leg PV
        pv_float = self._float_leg_pv(
            swap = swap,
            float_dates = floating_leg_dates,
            valuation_time = valuation_time
        )

        # compute swap PV
        pv_swap = pv_float - pv_fixed if swap.pay_fixed else pv_fixed - pv_float

        return pv_swap
    
    # risk analytics layer
    def exposure_profile(
            self,
            swap,
            steps_per_year: int = 2
    ):
        """ 
        Return the exposure profile for each cashflow date over the life of a trade 
        
        Exposure at future time t:
            E(t) = max(V_{t}, 0)

            where
                - V_{t}: mark-to-market future value at time t
        """
        cf_grid = CashflowGridder.cf_grid(
            maturity = swap.maturity,
            steps_per_year = steps_per_year
        )

        exposure_rows = []

        for t in cf_grid:

            pv = self._forward_pv(
                swap = swap,
                valuation_time = t
            )

            exposure = max(pv, 0.0)

            exposure_rows.append({
                'Time': t,
                'PV': pv,
                'Exposure': exposure
            })
        
        return pd.DataFrame(exposure_rows)
    
    def expected_exposure(
            self,
            swap,
            steps_per_year: int = 2
    ):
        """ 
        Return the expected exposure of a trade at any given future point in time
        
        Expected Exposure:
            EE_{t} = E[E_{t}]

        Under deterministic curve assumption:
            -> EE_{t} = E_{t}
           """
        profile = self.exposure_profile(
            swap = swap,
            steps_per_year = steps_per_year
        )

        # expected exposure
        profile['EE'] = profile['Exposure']

        # expected negative exposure
        profile['ENE'] = np.maximum(-profile['PV'], 0)

        return profile[['Time', 'EE', 'ENE']]
    

    # reporting layer
    def exposure_report(
            self,
            swap,
            steps_per_year: int = 2
    ):
        """ Summary exposure metrics """
        profile = self.exposure_profile(
            swap = swap,
            steps_per_year = steps_per_year
        )

        return pd.DataFrame({
            'Metric': [
                'Current Exposure',
                'Max Exposure',
                'Average Exposure'
            ],
            'Value': [
                profile['Exposure'].iloc[0],
                profile['Exposure'].max(),
                profile['Exposure'].mean()
            ]
        })