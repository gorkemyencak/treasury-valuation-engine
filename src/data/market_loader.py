import pandas as pd

from src.config.dataset_config import CURVE_CONFIG

from src.data.fred_downloader import FredCurveDownloader

class MarketLoader:
    """ Downloads, cleans and aligns dates of selected market dataset """
    def __init__(self):

        # attributes
        self.raw_curves = {}
        self.clean_curves = {}

    # download raw curves
    def download_curves(self):
        """ Download market dataset using FredCurveDownloader class """
        for curve in CURVE_CONFIG.keys():
            loader = FredCurveDownloader(
                curve_name = curve
            )

            self.raw_curves[curve] = loader.download()

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
    
    # clean all curves
    def clean_all_curves(self) -> None:
        """ Data cleansing step for all curves stored in self.raw_curves """
        for name, df_curve in self.raw_curves.items():

            self.clean_curves[name] = self._clean_curve(df = df_curve)
    
    # date alignment across all curves
    def align_dates(self):
        """ Aligning date index for all curves stored in self.clean_curves """
        df_merged = (
            pd.concat(
                self.clean_curves.values(),
                axis = 1,
                keys = self.clean_curves.keys()
            )
            .ffill()
            .dropna()
        )

        return df_merged
    
    # market loader pipeline
    def loader_pipeline(self): #-> pd.DataFrame:
        """ Market loader pipeline performing downloading, data cleansing and date alignment steps """
        self.download_curves()
        self.clean_all_curves()
        df_curves = self.align_dates()
        
        return df_curves