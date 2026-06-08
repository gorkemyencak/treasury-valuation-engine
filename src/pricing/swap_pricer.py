import pandas as pd

from src.schedules.payment_schedule import PaymentSchedule

from src.curves.discount_curve import DiscountCurve
from src.curves.projection_curve import ProjectionCurve

class SwapPricer:
    ''' 
    Interest rate swap valuation module 
    
    PV Formula:
        -> Pay-fixed swap 
            PV = PV_float - PV_fixed
        -> Receive-fixed swap
            PV = PV_fixed - PV_float
    
    PV (fixed-leg):
        PV_fixed = N * K * sum_{i=1,..,n} (alpha_{i} * DF(T_{i}))

        where 
            - N: notional amount
            - K: fixed swap rate
            - alpha_{i}: accrual fraction
            - DF(T_{i}): discount factor

    PV (floating-leg):
        PV_float = N * sum_{i=1,..,n} (f_{i} * alpha_{i} * DF(T_{i}))

        where
            - N: notional amount
            - f_{i}: forward rate (projection curve)
            - alpha_{i}: accrual fraction
            - DF(T_{i}): discount factor

    For a par swap:
        S = (1 - DF(T)) / sum_{i=1,..,n} (alpha_{i} * DF(T_{i}))

        where
            - S: fair fixed swap rate
            - alpha_{i}: accrual factor
            - DF(T_{i}): discount factor at time i 
    '''
    def __init__(
            self,
            discount_curve: DiscountCurve,
            projection_curve: ProjectionCurve
    ):
        
        # attributes
        self.discount_curve = discount_curve
        self.projection_curve = projection_curve

        self.freq_map = PaymentSchedule.FREQUENCY_MAP


    def _fixed_leg_pv(
            self,
            swap
    ) -> float:
        """ Compute fixed-leg PV """
        pv = 0.0

        # payment dates relative to valuation date
        aged_dates = swap.fixed_schedule.generate_schedule()

        # payment dates on original timeline
        original_dates = swap.fixed_schedule.generate_original_schedule()

        #if swap.valuation_time is not None:
        #    valuation_time = swap.valuation_time

        valuation_time = getattr(swap, 'valuation_time', 0)

        previous_date = valuation_time

        fixed_rate = swap.fixed_rate / 100.0

        for payment_date, discount_time in zip(original_dates, aged_dates):

            accrual = payment_date - previous_date

            df = self.discount_curve.get_discount_factor(maturity = discount_time)

            pv += accrual * df

            previous_date = payment_date
        
        pv_fixed = swap.notional * fixed_rate * pv

        return pv_fixed
    

    def _float_leg_pv(
            self,
            swap
    ) -> float:
        """ Compute floating-leg PV """
        pv = 0.0

        # payment dates relative to valuation date
        aged_dates = swap.float_schedule.generate_schedule() 

        # payment dates on original timeline
        original_dates = swap.float_schedule.generate_original_schedule()

        #if swap.valuation_time is not None:
        #    valuation_time = swap.valuation_time

        valuation_time = getattr(swap, 'valuation_time', 0)

        previous_date = valuation_time

        for payment_date, discount_time in zip(original_dates, aged_dates):

            accrual = payment_date - previous_date

            # compute forward rate
            forward_rate = self.projection_curve.get_forward_rate(
                start = previous_date,
                end = payment_date
            )

            df = self.discount_curve.get_discount_factor(maturity = discount_time)

            pv += forward_rate * accrual * df

            previous_date = payment_date

        pv_float = swap.notional * pv

        return pv_float
    

    def price(
            self,
            swap
    ) -> float:
        """ Return PV of an IR swap """
        # compute fixed and floating leg PVs
        pv_fixed = self._fixed_leg_pv(swap = swap)
        pv_float = self._float_leg_pv(swap = swap)

        # Pay-fixed IR swap
        return pv_float - pv_fixed if swap.pay_fixed == True else pv_fixed - pv_float
    

    def par_rate(
            self,
            swap
    ):
        """ Returns fair fixed swap rate of a par swap using fixed-leg payment dates """
        payment_dates = swap.fixed_schedule.generate_schedule()

        accrual = 1.0 / self.freq_map[swap.fixed_freq]

        annuity = 0.0

        for t in payment_dates:

            df = self.discount_curve.get_discount_factor(maturity = t)

            annuity += accrual * df
        
        # discount factor at maturity
        maturity_df = self.discount_curve.get_discount_factor(maturity = swap.maturity)

        # compute par rate
        swap_par_rate = (1.0 - maturity_df) / annuity

        return swap_par_rate

    
    def valuation_report(
            self,
            swap
    ):
        """ Comparing fixed- vs floating-leg of an IR swap """
        # compute fixed and floating leg PVs
        pv_fixed = self._fixed_leg_pv(swap = swap)
        pv_float = self._float_leg_pv(swap = swap)

        # compute PV
        PV = self.price(swap = swap)

        return pd.DataFrame({
            'Metric': [
                'Fixed-leg PV',
                'Floating-leg PV',
                'Swap PV'
            ],
            'Value': [
                pv_fixed,
                pv_float,
                PV
            ]
        })