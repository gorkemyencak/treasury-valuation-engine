import pandas as pd

from src.schedules.payment_schedule import PaymentSchedule

from src.curves.simulator.hull_white_2factor_pricer import HullWhite2FactorPricer

class HullWhite2FactorSwapPricer:
    """ 
    Hull-White 2-Factor Interest Rate Swap Pricer module
    
    This module uses HullWhite2FactorPricer to price an IRS using HW 2Factor bond prices P(t, T) generated from factor realizations x_{t} and y_{t}
    """
    def __init__(
            self,
            hw_pricer: HullWhite2FactorPricer
    ):
        # attributes
        self.hw_pricer = hw_pricer

    # helper functions
    def _fixed_leg_pv(
            self,
            swap,
            x_t: float,
            y_t: float
    ) -> float:
        """  
        Computes fixed-leg PV

        Formula:
            PV_fixed = N * K * sum_{i=1,..,n} (alpha_{i} * P(t, T_{i}))

            where
                - N: notional amount
                - K: fixed swap rate
                - alpha_{i}: accrual fraction
                - P(t, T_{i}): bond price at valuation time t of a zero-coupon maturing at time T_{i}  
        """
        pv = 0.0

        # valuation time
        valuation_time = getattr(swap, 'valuation_time', 0)

        # payment dates on original timeline
        payment_dates = swap.fixed_schedule.generate_original_schedule()

        fixed_rate = swap.fixed_rate / 100.0

        previous_date = valuation_time

        for payment_date in payment_dates:

            accrual = payment_date - previous_date

            df = self.hw_pricer.bond_price(
                t = valuation_time,
                T = payment_date,
                x_t = x_t,
                y_t = y_t
            )

            pv += accrual * df

            previous_date = payment_date
        
        pv_fixed = swap.notional * fixed_rate * pv

        return pv_fixed
    
    def _float_leg_pv(
            self,
            swap,
            x_t: float,
            y_t: float
    ) -> float:
        """  
        Computes float-leg PV

        Formula:
            PV_float = N * sum_{i=1,..,n} (F_{i} * alpha_{i} * P(t, T_{i}))

            where
                - N: notional amount
                - F_{i}: forward rate -> F_{i} = ( P(t, T_{i-1}) / P(t, T_{i}) - 1) / alpha_{i}
                - alpha_{i}: accrual fraction
                - P(t, T_{i}): bond price at valuation time t of a zero-coupon maturing at time T_{i}  
        """
        pv = 0.0

        # valuation time
        valuation_time = getattr(swap, 'valuation_time', 0)

        # payment dates on original timeline
        payment_dates = swap.float_schedule.generate_original_schedule()

        previous_date = valuation_time

        for payment_date in payment_dates:

            accrual = payment_date - previous_date

            # compute forward rate
            P_t_T0 = self.hw_pricer.bond_price(
                t = valuation_time,
                T = previous_date,
                x_t = x_t,
                y_t = y_t
            )

            P_t_T1 = self.hw_pricer.bond_price(
                t = valuation_time,
                T = payment_date,
                x_t = x_t,
                y_t = y_t
            )

            forward_rate = (P_t_T0 / P_t_T1 - 1) / accrual

            pv += forward_rate * accrual * P_t_T1

            previous_date = payment_date
        
        pv_float = swap.notional * pv

        return pv_float
    
    # risk analytics layer
    def price(
            self,
            swap,
            x_t,
            y_t
    ):
        """
        Return PV of an IR Swap using HW2Factor Swap Pricer bond pricing logic

        PV Formula:
            -> Pay-fixed swap 
                PV = PV_float - PV_fixed
            -> Receive-fixed swap
                PV = PV_fixed - PV_float
        """
        # fixed-leg PV
        pv_fixed = self._fixed_leg_pv(
            swap = swap,
            x_t = x_t,
            y_t = y_t
        )

        # float-leg PV
        pv_float = self._float_leg_pv(
            swap = swap,
            x_t = x_t,
            y_t = y_t
        )

        return pv_float - pv_fixed if swap.pay_fixed else pv_fixed - pv_float