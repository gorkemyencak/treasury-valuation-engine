import pandas as pd
import numpy as np

from src.utils.parser import TenorParser

class CurveSnapshot:
    """ Immutable market curve snapshot at a signle valuation date """
    def __init__(
            self,
            curve_name: str,
            as_of_date: pd.Timestamp,
            tenors: list[str],
            rates
    ):
        # attributes
        self._curve_name = curve_name
        self._as_of_date = as_of_date
        self._tenors = tenors
        self._rates = np.array(rates, dtype = float)

        self._tenor_years = np.array([
            TenorParser.tenors_to_years(tenor = t)
            for t in self._tenors
        ])

        self._validate()


    def _validate(self):

        if len(self._tenors) != len(self._rates):

            raise ValueError('Tenor and Rate length mismatch')
        
        if len(self._tenor_years) != len(self._rates):

            raise ValueError('Year Fraction and Rate length mismatch')
    
    @classmethod
    def snap_from_df_row(
        cls,
        curve_name: str,
        as_of_date: pd.Timestamp,
        curve_row: pd.Series
    ):
        """ Creating curve snapshot from dataframe row """
        curve_row = curve_row.dropna()
        tenors = list(curve_row.index)
        rates = curve_row.values.astype(float)

        return cls(
            curve_name = curve_name,
            as_of_date = as_of_date,
            tenors = tenors,
            rates = rates
        )
    
    @property
    def curve_name(self):
        return self._curve_name
    
    @property
    def as_of_date(self):
        return self._as_of_date
    
    @property
    def tenors(self):
        return self._tenors
    
    @property
    def rates(self):
        return self._rates
    
    @property
    def tenor_years(self):
        return self._tenor_years
    
    
    def summary(self) -> pd.DataFrame:

        return pd.DataFrame({
            'Tenor': self._tenors,
            'Years': self._tenor_years,
            'Rate': self._rates
        })
    

    def __repr__(self):

        return (
            f'CurveSnapshot('
            f'curve_name={self._curve_name}, '
            f'as_of_date={self._as_of_date}, '
            f'tenor_length={len(self._tenors)}'
            f')'
        )