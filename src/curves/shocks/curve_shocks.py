import numpy as np
import pandas as pd
from copy import deepcopy

from src.curves.base_curve import BaseCurve
from src.curves.zero_curve import ZeroCurve

class CurveShockEngine:
    """
    Basel compliant IR curve shock framework
    """

    @staticmethod
    def parallel_shift(
        curve: ZeroCurve,
        parallel_shock_in_bps: int
    ):
        """ Parallel shift entire curve by parallel_shock_in_bps """
        shocked_curve = deepcopy(curve)

        parallel_shock = parallel_shock_in_bps / 10000.0

        #shocked_curve.zero_rates += parallel_shock
        for m, rate in shocked_curve.zero_rates.items():

            shocked_curve.update_zero_rate(
                maturity = m,
                new_rate = rate + parallel_shock
            )

        return shocked_curve
    

    @staticmethod
    def key_rate_shift(
        curve: ZeroCurve,
        maturity: float,
        shock_in_bps: int
    ):
        """ Key rate shock applied to nearest maturity """
        shocked_curve = deepcopy(curve)

        shock = shock_in_bps / 10000.0

        maturities = np.array(list(shocked_curve.zero_rates.keys()))

        nearest_idx = np.argmin(
            np.abs(
                maturities - maturity
            )
        )

        nearest_maturity = maturities[nearest_idx]

        shocked_curve.update_zero_rate(
            maturity = nearest_maturity,
            new_rate = shocked_curve.zero_rates[nearest_maturity] + shock
        )
        
        return shocked_curve
    
    @staticmethod
    def steepener(
        curve: ZeroCurve,
        short_in_bps: int = -20,
        long_in_bps: int = +20
    ):
        """ Curve steepener -> short-end down, long-end up """
        if short_in_bps > long_in_bps:
            raise ValueError('Long rate must be higher than short rate in curve steepener shock')

        shocked_curve = deepcopy(curve)

        short_shock = short_in_bps / 10000.0
        long_shock = long_in_bps / 10000.0

        maturities = np.array(list(shocked_curve.zero_rates.keys()))

        weights = (maturities - maturities.min()) / (maturities.max() - maturities.min())

        for maturity, weight in zip(maturities, weights):

            shock = short_shock + (long_shock - short_shock) * weight

            shocked_curve.update_zero_rate(
                maturity = maturity,
                new_rate = shocked_curve.zero_rates[maturity] + shock
            )

        return shocked_curve
    

    @staticmethod
    def flattener(
        curve: ZeroCurve,
        short_in_bps: int = +20,
        long_in_bps: int = -20
    ):
        """ Curve flattener -> short-end up, long-end down """
        if long_in_bps > short_in_bps:
            raise ValueError('Short rate must be higher than short rate in curve flattener shock')
        
        shocked_curve = deepcopy(curve)

        short_shock = short_in_bps / 10000.0
        long_shock = long_in_bps / 10000.0

        maturities = np.array(list(shocked_curve.zero_rates.keys()))

        weights = (maturities - maturities.min()) / (maturities.max() - maturities.min())

        for maturity, weight in zip(maturities, weights):

            shock = short_shock + (long_shock - short_shock) * weight

            shocked_curve.update_zero_rate(
                maturity = maturity,
                new_rate = shocked_curve.zero_rates[maturity] + shock
            )

        return shocked_curve

    
    @staticmethod
    def shock_report(
        curve: ZeroCurve,
        shocked_curve
    ) -> pd.DataFrame:
        """ Returns a shock report comparing base and shocked curve as well as the curve shock in bps """
        maturities = list(curve.zero_rates.keys())
        
        return pd.DataFrame({
            'Maturity': maturities,
            'BaseRate': [
                curve.zero_rates[m] * 100
                for m in maturities
            ],
            'ShockedRate': [
                shocked_curve.zero_rates[m] * 100
                for m in maturities
            ],
            'ShockBps': [
                (shocked_curve.zero_rates[m] - curve.zero_rates[m]) * 10000
                for m in maturities
            ]
        })