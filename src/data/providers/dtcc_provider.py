import pandas as pd
import requests

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
        self.metadata_path = self.file_path.parent / f'{self.curve_name}_metadata.csv'


    def _fetch_raw_data(self) -> pd.DataFrame:
        """ Fetching OIS curve from JSON response """
        url = self.config['api_endpoint']

        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        
        # fetching ois data from json response
        if json_data.get('currency') != self.config['currency']:
            raise ValueError('Currency mismatch in DTCC response')
        
        ois_curve = json_data['curve']

        return pd.DataFrame(ois_curve)
    

    def _transform_curve(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ 
        Transforming the DTCC swap curve into base curve format that is consistent with other market curve data 
        
        Metadata will be separately stored that might be useful in later steps of the project, i.e. the number of trades as a liquidity proxy 
        """
        # curve data -> maintaining the consistent data structure with other market curves
        curve_data = {}

        # metadata 
        metadata_rows = []

        for _, row in df.iterrows():

            tenor = row['tenor']
            curve_data[tenor] = row['rate']

            metadata_rows.append({
                'tenor': tenor,
                'days': row['days'],
                'trades': row['trades'],
                'method': row['method'],
                'source': row['source']                
            })

        self.metadata = pd.DataFrame(metadata_rows)

        curve_df = pd.DataFrame(
            [curve_data],
            index = [pd.to_datetime(df['date'].iloc[0])]
        )

        curve_df.index.name = 'Date'

        return curve_df
    

    def download(self) -> pd.DataFrame:
        """ Download specified curve from DTCC, return exception if unavailable, and save locally """
        if not self.file_path.exists():
            print(f'Downloading {self.curve_name} cure from DTCC..')
            raw_df = self._fetch_raw_data()

            curve_df = self._transform_curve(df = raw_df)

            # save the dataframe locally
            curve_df.to_csv(self.file_path)

            # save metadata locally
            if self.metadata is not None:
                self.metadata.to_csv(
                    self.metadata_path,
                    index = False
                )

            return curve_df
        
        else:
            print(f'{self.curve_name} curve dataset already downloaded..')

            return pd.read_csv(
                self.file_path,
                index_col = 0,
                parse_dates = True
            )