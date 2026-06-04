import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from src.curves.curve_snapshot import CurveSnapshot
from src.curves.base_curve import BaseCurve

class DiscountCurve(BaseCurve):
    """ Discount factor curve container storing discount factors and provides interpolation utilities """
    def __init__(
            self, 
            curve_snapshot: CurveSnapshot, 
            maturities: list[float],
            discount_factors: list[float],
            interpolation_method: str = 'linear'
    ):
        
        # attributes
        self.snapshot = curve_snapshot
        self.interpolation_method = interpolation_method

        self._maturities = np.array(
            maturities,
            dtype = float
        )

        self._discount_factors = np.array(
            discount_factors,
            dtype = float
        )

        super().__init__(
            curve_snapshot = self.snapshot, 
            interpolation_method = self.interpolation_method
        )

        self._build_df_interpolator()
    
    @property
    def maturities(self):
        return self._maturities
    
    @property
    def discount_factors(self):
        return self._discount_factors
    
    def _build_df_interpolator(self):
        """ Interpolate discount curve w.r.t. interpolation method """
        self.df_interpolator = interp1d(
            x = self._maturities,
            y = self._discount_factors,
            kind = self.interpolation_method,
            fill_value = 'extrapolate'      # type: ignore
        )
    
    def get_discount_factor(
            self,
            maturity
    ):
        """ Returns the discount factor of a given maturity """
        return float(self.df_interpolator(maturity))
    
    
    def summary(self) -> pd.DataFrame:

        return pd.DataFrame({
            'Maturity': self._maturities,
            'DiscountFactor': self._discount_factors
        })
    

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'curve_name={self.curve_name}, '
            f'tenor_length={len(self._maturities)}'
            f')'
        )