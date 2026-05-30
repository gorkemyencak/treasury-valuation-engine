import pandas as pd

from src.config.dataset_config import DTCC_CONFIG

from src.data.providers.base_provider import BaseMarketDataProvider

class DTCCCurveProvider(BaseMarketDataProvider):
    """ Downloading DTCC swap curves from DTCC data portal """
    def __init__(
            self, 
            curve_name: str, 
            data_dir: str = 'data/swaps'
    ):
        # superclass initializer
        super().__init__(
            curve_name = curve_name,
            data_dir = data_dir
        )

        if curve_name not in DTCC_CONFIG:
            raise ValueError(f'Unknown DTCC curve: {curve_name}')
        
        # attributes
        self.config = DTCC_CONFIG[curve_name]
        self.metadata = None


    def _fetch_raw_data(self):
        """ Temporary sample data for development purpose """
        # raise NotImplementedError
        return pd.DataFrame({
            'tenor': ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y'],
            'rate':  [3.82, 3.86, 3.85, 3.86, 3.87, 3.93,  4.03,  4.21,  4.28,  4.26],
            'df':    [0.962,0.925,0.891,0.857,0.824,0.759,0.668, 0.532, 0.425, 0.278],
            'trades':[  48,  108,   58,   26,  252,   81,   323,    22,    44,   161]
        })
    

    def _transform_curve(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ 
        Transforming the DTCC swap curve into base curve format that is consistent with other market curve data 
        
        Metadata will be separately stored that might be useful in later steps of the project, i.e. the number of trades as a liquidity proxy 
        """
        curve_data = {}

        # metadata 
        metadata_rows = []
    

    def download(self) -> pd.DataFrame:
        """ Download specified curve from DTCC, return exception if unavailable, and save locally """
        if not self.file_path.exists():
            print(f'Downloading {self.curve_name} cure from DTCC..')
            raw_df = self._fetch_raw_data()

            curve_df = self._normalize_curve(df = raw_df)

            # save the dataframe locally
            curve_df.to_csv(self.file_path)

            return curve_df
        
        else:
            print(f'{self.curve_name} curve dataset already downloaded..')

            return pd.read_csv(
                self.file_path,
                index_col = 0,
                parse_dates = True
            )