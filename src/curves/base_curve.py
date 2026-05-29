import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.curves.curve_snapshot import CurveSnapshot

class BaseCurve:
    """ Continuous interpolated curve object """
    def __init__(
            self,
            curve_snapshot: CurveSnapshot,
            interpolation_method: str = 'linear'
    ):
        # attributes
        self.snapshot = curve_snapshot
        self.interpolation_method = interpolation_method

        self.curve_name = self.snapshot.curve_name
        self.as_of_date = self.snapshot.as_of_date
        self.tenors = self.snapshot.tenors
        self.x = self.snapshot.tenor_years
        self.y = self.snapshot.rates / 100

        self._build_interpolator()


    def _build_interpolator(self):
        """ Interpolate curve w.r.t. interpolation method """
        # curve interpolator
        self.interpolator = interp1d(
            x = self.x,
            y = self.y,
            kind = self.interpolation_method,
            fill_value = 'extrapolate'      # type: ignore
        )  

    
    def curve_points(self):

        return {
            'maturities': self.x,
            'rates': self.y
        }
    

    def __repr__(self):

        return (
            f'BaseCurve('
            f'curve_name={self.curve_name}, '
            f'as_of_date={self.as_of_date}, '
            f'tenor_length={len(self.x)}, '
            f'interpolation={self.interpolation_method}'
            f')'
        )