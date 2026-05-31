import pandas as pd

from src.config.dataset_config import (
    FRED_CONFIG,
    ECB_CONFIG,
    DTCC_CONFIG
)

from src.data.providers.fred_provider import FredCurveProvider
from src.data.providers.ecb_provider import ECBCurveProvider
from src.data.providers.dtcc_provider import DTCCCurveProvider

class MarketLoader:
    """ Downloads, cleans and aligns dates of selected market dataset """
    def __init__(self):

        # attributes
        self.raw_market_curves = {}
        self.raw_swap_curves = {}

        self.clean_market_curves = {}
        self.clean_swap_curves = {}

    # download historical raw market curves
    def download_market_curves(self):
        """ Download historical market dataset using BaseMarketDataProvider abstract class """
        # FRED
        for curve in FRED_CONFIG.keys():
            loader = FredCurveProvider(
                curve_name = curve
            )

            self.raw_market_curves[curve] = loader.download()
        
        # ECB
        for curve in ECB_CONFIG.keys():
            loader = ECBCurveProvider(
                curve_name = curve
            )

            self.raw_market_curves[curve] = loader.download()

    # download swap raw market curves
    def download_swap_curves(self):
        
        # DTCC
        for curve in DTCC_CONFIG.keys():
            loader = DTCCCurveProvider(
                curve_name = curve
            )
        
            self.raw_swap_curves[curve] = loader.download()

    # clean single curve
    def _clean_curve(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Data cleansing step for a single curve """
        df = df.copy()

        # convert to numeric columns
        df = df.apply(
            pd.to_numeric,
            errors = 'coerce'
        )

        # handling missing values (forward fill)
        df = df.ffill()

        # dropping empty rows
        df = df.dropna(how = 'all')
        
        return df
    
    # clean historical market curves
    def clean_market(self) -> None:
        """ Data cleansing step for all historical market curves stored in self.raw_market_curves """
        for name, df_curve in self.raw_market_curves.items():
            self.clean_market_curves[name] = self._clean_curve(df = df_curve)

    # clean swap curves
    def clean_swap(self) -> None:
        """ Data cleansing step for all swap curves stored in self.raw_swap_curves """
        for name, df_curve in self.raw_swap_curves.items():
            self.clean_swap_curves[name] = self._clean_curve(df = df_curve)
    
    # date alignment across all curves
    def align_dates(
            self,
            curves_dict: dict
    ):
        """ Aligning date index for all curves stored in self.clean_curves """
        df_merged = (
            pd.concat(
                curves_dict.values(),
                axis = 1,
                keys = curves_dict.keys()
            )
            .ffill()
            .dropna()
        )

        return df_merged
    
    # historical market loader pipeline
    def market_loader_pipeline(self):
        """ Historical market curve loader pipeline performing downloading, data cleansing and date alignment steps """
        self.download_market_curves()
        self.clean_market()
        df_market_curves = self.align_dates(curves_dict = self.clean_market_curves)
        
        return df_market_curves
    
    # swap loader pipeline
    def swap_loader_pipeline(self):
        """ Swap curve loader pipeline performing downloading and data cleansing steps """
        self.download_swap_curves()
        self.clean_swap()
        df_swap_curves = self.align_dates(curves_dict = self.clean_swap_curves)

        return df_swap_curves