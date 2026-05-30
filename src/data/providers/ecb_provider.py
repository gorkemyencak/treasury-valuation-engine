import pandas as pd
import requests
from io import StringIO

from src.config.dataset_config import ECB_CONFIG

from src.data.providers.base_provider import BaseMarketDataProvider


BASE_URL = 'https://data-api.ecb.europa.eu/service/data'

class ECBCurveProvider(BaseMarketDataProvider):
    """ Downloading ECB market curves from ECB Data Portal """
    def __init__(
            self, 
            curve_name: str, 
            data_dir: str = 'data/curves'
    ):
        # superclass initializer
        super().__init__(
            curve_name = curve_name,
            data_dir = data_dir
        )

        if curve_name not in ECB_CONFIG:
            raise ValueError(f'Unknown ECB curve: {curve_name}')
        
        # attributes
        self.series_map = ECB_CONFIG[self.curve_name]


    def _fetch_series(
            self,
            flow: str,
            key: str
    ) -> pd.Series:
        """ Fetching specified flow/key series from ECB portal """
        url = f'{BASE_URL}/{flow}/{key}'

        params = {
            'format': 'csvdata'
        }

        headers = {
            'Accept': 'text/csv'
        }

        response = requests.get(
            url,
            params = params,
            headers = headers,
            timeout = 30
        )

        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))

        # ECB standard columns
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'])
        df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors = 'coerce')

        df = df.set_index('TIME_PERIOD')

        return df['OBS_VALUE']

    
    def download(self) -> pd.DataFrame:
        """ Download specified curve from ECB portal, return exception if unavailable, and save locally """
        if not self.file_path.exists():
            print(f'Downloading{self.curve_name} curve from ECB')
            df = pd.DataFrame()

            for tenor, series in self.series_map.items():
                try:
                    splitted_series_key = series.split('.')

                    flow = splitted_series_key[0]

                    key = '.'.join(splitted_series_key[1:])

                    df[tenor] = self._fetch_series(
                        flow = flow,
                        key = key
                    )

                except Exception as e:
                    print(f'{series} not available -> {e}')
                    df[tenor] = None
                
            df.index.name = 'Date'
            df = df.sort_index()

            # save the dataframe locally
            df.to_csv(self.file_path)

            return df
        
        else:
            print(f'{self.curve_name} curve dataset already downloaded..')

            return pd.read_csv(
                self.file_path,
                index_col = 0,
                parse_dates = True
            )