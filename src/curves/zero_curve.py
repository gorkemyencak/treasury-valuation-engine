import numpy as np
import pandas as pd

from src.curves.discount_curve import DiscountCurve

class ZeroCurve:
    """ Continuously compounded zero curve """
    def __init__(
            self,
            discount_curve: DiscountCurve
    ):
        
        # attributes
        self.discount_curve = discount_curve
        self.maturities = self.discount_curve.maturities
        self.dfs = self.discount_curve.discount_factors

        self.zero_rates = self._build_zero_curve()

    def _build_zero_curve(self) -> dict:
        """ 
        Formula:
            z(t) = - ln(DF(t)) / t
        """
        zero_curve = {}

        for t, df in zip(self.maturities, self.dfs):

            if t == 0:
                continue

            zero_rate = -np.log(df) / t

            zero_curve[t] = float(zero_rate)

        return zero_curve
    

    def get_zero_rate(
            self,
            maturity
    ):
        """ Returns zero-rate of a given maturity """
        return self.zero_rates[maturity]
    
    def summary(self) -> pd.DataFrame:

        return pd.DataFrame({
            'Maturity': t,
            'ZeroRate': r * 100
        } for t, r in self.zero_rates.items()
        )


