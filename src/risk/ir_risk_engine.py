import numpy as np
import pandas as pd

from src.pricing.swap_pricer import SwapPricer

from src.curves.zero_curve import ZeroCurve
from src.curves.shocks.curve_shocks import CurveShockEngine

class IRRiskEngine:
    """ 
    IR risk analytics 
    
    Allows to compute:
        - PV01
        - DV01
        - Basel IRRBB shock integration
    """
    def __init__(
            self,
            pricer: SwapPricer,
            zero_curve: ZeroCurve
    ):
        
        # attributes
        self.pricer = pricer
        self.zero_curve = zero_curve

    # helper for repricing swaps w.r.t. shocked discount curve
    def _reprice(
            self,
            swap,
            discount_curve = None
    ):
        """ Reprice swap using shocked discount curve, reprice base PV if no discount curve is provided """
        if discount_curve is None:

            return self.pricer.price(swap = swap)
        
        # shocked pricer
        shocked_pricer = SwapPricer(
            discount_curve = discount_curve,
            projection_curve = self.pricer.projection_curve
        )

        return shocked_pricer.price(swap = swap)
    
    
    # risk analytics layer
    def pv01(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> float:
        """ 
        PV01 for a +1bp parallel move in rates 
        
        PV01 Formula:
            PV01 = PV(r + 1bp) - PV(r)

            where
                - PV(r): current swap value
                - PV(r + 1bp): shocked swap value
        """
        # compute PV_base
        pv_base = self._reprice(swap = swap)

        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = shock_in_bps
        )

        # convert shocked zero curve -> discount curve
        shocked_dc = shocked_zero_curve.to_discount_curve()

        # shocked PV
        pv_shock = self._reprice(
            swap = swap,
            discount_curve = shocked_dc
        )

        return pv_shock - pv_base
    

    def dv01(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> float:
        """
        DV01 for a +1bp move in rates

        DV01 Formula:
            DV01 = -PV01
        """
        return -self.pv01(
            swap = swap,
            shock_in_bps = shock_in_bps
        )
    
    
    def dv01_central(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> float:
        """
        Central difference DV01

        Formula:
            DV01_{central} = -[PV(r + 1bp) - PV(r - 1bp)] / 2
        """
        ### upward shock
        # shocked zero curve
        up_shock_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = shock_in_bps
        )

        # shocked discount curve
        up_shocked_dc = up_shock_zero_curve.to_discount_curve()

        ### downward shock
        # shocked zero curve
        down_shock_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = -shock_in_bps
        )

        # shocked discount curve
        down_shocked_dc = down_shock_zero_curve.to_discount_curve()

        # compute pv upward shock
        pv_shock_up = self._reprice(
            swap = swap,
            discount_curve = up_shocked_dc
        )

        # compute pv downward shock
        pv_shock_down = self._reprice(
            swap = swap,
            discount_curve = down_shocked_dc
        )

        return -(pv_shock_up - pv_shock_down) / 2.0
    

    # reporting layer
    def ir_risk_report(
            self,
            swap,
            shock_in_bps: int = 1
    ):
        """ Return IR risk metrics of a swap """
        return pd.DataFrame({
            'Metric': [
                'PV',
                'PV01',
                'DV01',
                'Central DV01'
            ],
            'Value': [
                self.pricer.price(swap = swap),
                self.pv01(swap = swap, shock_in_bps = shock_in_bps),
                self.dv01(swap = swap, shock_in_bps = shock_in_bps),
                self.dv01_central(swap = swap, shock_in_bps = shock_in_bps)
            ]
        })