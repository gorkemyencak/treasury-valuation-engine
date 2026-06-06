import numpy as np
import pandas as pd

from src.pricing.swap_pricer import SwapPricer

from src.curves.zero_curve import ZeroCurve
from src.curves.projection_curve import ProjectionCurve
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

    # helper functions
    def _reprice(
            self,
            swap,
            discount_curve = None,
            projection_curve = None
    ):
        """ Reprice swap using shocked discount curve, reprice base PV if no discount curve is provided """
        if discount_curve is None:
            discount_curve = self.pricer.discount_curve

        if projection_curve is None:
            projection_curve = self.pricer.projection_curve
        
        # shocked pricer
        shocked_pricer = SwapPricer(
            discount_curve = discount_curve,
            projection_curve = projection_curve
        )

        return shocked_pricer.price(swap = swap)
    
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
    
    def _scenario_dv01(
            self,
            swap,
            shocked_zero_curve
    ):
        """ Generic scenario DV01 """
        # compute base PV
        pv_base = self._reprice(swap = swap)

        # convert shocked zero curve -> discount curve, projection curve
        shocked_dc, shocked_proj = self._build_shocked_curves(shocked_zero_curve = shocked_zero_curve)

        # shocked PV
        pv_shock = self._reprice(
            swap = swap,
            discount_curve = shocked_dc,
            projection_curve = shocked_proj
        )

        return -(pv_shock - pv_base)
    
    
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
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = shock_in_bps
        )

        return -self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
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

        # shocked discount & projection curve
        up_shocked_dc, up_shocked_proj = self._build_shocked_curves(shocked_zero_curve = up_shock_zero_curve)

        ### downward shock
        # shocked zero curve
        down_shock_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = -shock_in_bps
        )

        # shocked discount & projection curve
        down_shocked_dc, down_shocked_proj = self._build_shocked_curves(shocked_zero_curve = down_shock_zero_curve)

        # compute pv upward shock
        pv_shock_up = self._reprice(
            swap = swap,
            discount_curve = up_shocked_dc,
            projection_curve = up_shocked_proj
        )

        # compute pv downward shock
        pv_shock_down = self._reprice(
            swap = swap,
            discount_curve = down_shocked_dc,
            projection_curve = down_shocked_proj
        )

        return -(pv_shock_up - pv_shock_down) / 2.0
    
    def key_rate_dv01(
            self,
            swap,
            maturity: float,
            shock_in_bps: int = 1
    ):
        """ Key-rate DV01 for a +1bp move in rates at a given tenor in the term structure """
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.key_rate_shift(
            curve = self.zero_curve,
            maturity = maturity,
            shock_in_bps = shock_in_bps
        )

        return self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
    def steepener_dv01(
            self,
            swap,
            short_shock_in_bps: int = -20,
            long_shock_in_bps: int = +20
    ):
        """ Curve steepener scenario DV01 """
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.steepener(
            curve = self.zero_curve,
            short_in_bps = short_shock_in_bps,
            long_in_bps = long_shock_in_bps
        )

        return self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
    def flattener_dv01(
            self,
            swap,
            short_shock_in_bps: int = +20,
            long_shock_in_bps: int = -20
    ):
        """ Curve flattener scenario DV01 """
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.flattener(
            curve = self.zero_curve,
            short_in_bps = short_shock_in_bps,
            long_in_bps = long_shock_in_bps
        )

        return self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
    def short_rate_up_dv01(
            self,
            swap,
            max_shock_in_bps: int = 100
    ):
        """ Short rate bump scenario DV01 """
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.short_rate_up(
            curve = self.zero_curve,
            max_shock_in_bps = max_shock_in_bps
        )

        return self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
    def short_rate_down_dv01(
            self,
            swap,
            max_shock_in_bps: int = -100
    ):
        """ Short rate drop scenario DV01 """
        # shocked zero curve
        shocked_zero_curve = CurveShockEngine.short_rate_down(
            curve = self.zero_curve,
            max_shock_in_bps = max_shock_in_bps
        )

        return self._scenario_dv01(
            swap = swap,
            shocked_zero_curve = shocked_zero_curve
        )
    
    def convexity(
            self,
            swap,
            shock_in_bps: int = 1
    ):
        """ 
        Non-linear extension to swap valuation 
        
        Convexity Formula:
            Convexity = [PV(r + Δr) - 2 * PV(r) + PV(r - Δr)] / (Δr)^2
        """
        # compute base PV
        pv_base = self._reprice(swap = swap)

        ### upward shock
        # shocked zero curve
        up_shock_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = shock_in_bps
        )

        # shocked discount & projection curve
        up_shocked_dc, up_shocked_proj = self._build_shocked_curves(shocked_zero_curve = up_shock_zero_curve)

        ### downward shock
        # shocked zero curve
        down_shock_zero_curve = CurveShockEngine.parallel_shift(
            curve = self.zero_curve,
            parallel_shock_in_bps = -shock_in_bps
        )

        # shocked discount & projection curve
        down_shocked_dc, down_shocked_proj = self._build_shocked_curves(shocked_zero_curve = down_shock_zero_curve)

        # compute pv upward shock
        pv_shock_up = self._reprice(
            swap = swap,
            discount_curve = up_shocked_dc,
            projection_curve = up_shocked_proj
        )

        # compute pv downward shock
        pv_shock_down = self._reprice(
            swap = swap,
            discount_curve = down_shocked_dc,
            projection_curve = down_shocked_proj
        )

        # convert shock_in_bps into decimal
        shock = shock_in_bps / 10000.0        

        return (pv_shock_up - 2 * pv_base + pv_shock_down) / (shock ** 2)


    # reporting layer
    def ir_risk_report(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> pd.DataFrame:
        """ Return IR risk metrics of a swap """
        return pd.DataFrame({
            'Metric': [
                'PV',
                'PV01',
                'DV01',
                'Central DV01',
                'Convexity'
            ],
            'Value': [
                self.pricer.price(swap = swap),
                self.pv01(swap = swap, shock_in_bps = shock_in_bps),
                self.dv01(swap = swap, shock_in_bps = shock_in_bps),
                self.dv01_central(swap = swap, shock_in_bps = shock_in_bps),
                self.convexity(swap = swap, shock_in_bps = shock_in_bps)
            ]
        })
    
    def key_rate_report(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> pd.DataFrame:
        """ Return key-rate DV01 of a swap for all key-rate tenors """

        key_rates = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0 ]

        return pd.DataFrame({
            'Maturity': key_rates,
            'KRDV01': [
                self.key_rate_dv01(
                    swap = swap,
                    maturity = m,
                    shock_in_bps = shock_in_bps
                )
                for m in key_rates
            ]
        })
    
    def scenario_report(
            self,
            swap,
            steepener_short_shock_in_bps: int = -20,
            steepener_long_shock_in_bps: int = +20,
            flattener_short_shock_in_bps: int = +20,
            flattener_long_shock_in_bps: int = -20
    ) -> pd.DataFrame:
        """ Return curve steepener/flattener scenario DV01 of a swap """
        return pd.DataFrame({
            'Scenario': [
                'Steepener',
                'Flattener'
            ],
            'DV01': [
                self.steepener_dv01(swap = swap, short_shock_in_bps = steepener_short_shock_in_bps, long_shock_in_bps = steepener_long_shock_in_bps),
                self.flattener_dv01(swap = swap, short_shock_in_bps = flattener_short_shock_in_bps, long_shock_in_bps = flattener_long_shock_in_bps)
            ]
        })
    
    def basel_irrbb_report(
            self,
            swap
    ) -> pd.DataFrame:
        # Basel shock scenarios
        parallel_up = self.dv01(swap = swap, shock_in_bps = 100)
        parallel_down = self.dv01(swap = swap, shock_in_bps = -100)
        steepener = self.steepener_dv01(swap = swap, short_shock_in_bps = -60, long_shock_in_bps = 60)
        flattener = self.flattener_dv01(swap = swap, short_shock_in_bps = 60, long_shock_in_bps = -60)
        short_rate_up = self.short_rate_up_dv01(swap = swap)
        short_rate_down = self.short_rate_down_dv01(swap = swap)

        return pd.DataFrame({
            'Scenario': [
                'Parallel Up',
                'Parallel Down',
                'Steepener',
                'Flattener',
                'Short Rate Up',
                'Short Rate Down'
            ],
            'DV01': [
                parallel_up,
                parallel_down,
                steepener,
                flattener,
                short_rate_up,
                short_rate_down
            ]
        })
    
    def dv01_recon_report(
            self,
            swap,
            shock_in_bps: int = 1
    ) -> pd.DataFrame:
        """ DV01 reconciliation report comparing key-rate sv01 and parallel dv01 of a swap """
        key_rate_report = self.key_rate_report(
            swap = swap,
            shock_in_bps = shock_in_bps
        )

        key_rate_sum = key_rate_report['KRDV01'].sum()

        parallel_dv01 = self.dv01(
            swap = swap,
            shock_in_bps = shock_in_bps
        )

        return pd.DataFrame({
            'Metric': [
                'Parallel DV01',
                'Sum KRDV01',
                'Difference'
            ],
            'Value': [
                parallel_dv01,
                key_rate_sum,
                parallel_dv01 - key_rate_sum
            ]
        })