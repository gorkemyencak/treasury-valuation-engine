import pandas as pd
from pathlib import Path

from abc import ABC, abstractmethod

class BaseMarketDataProvider(ABC):
    """ Abstract base class for all market data providers """
    def __init__(
            self,
            curve_name: str,
            data_dir: str = 'data/curves'
    ):
        # attributes
        self.curve_name = curve_name

        project_root = Path(__file__).resolve().parents[3]
        self.data_dir = project_root / data_dir
        self.data_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        self.file_path = self.data_dir / f'{self.curve_name}.csv'

    @abstractmethod
    def download(self) -> pd.DataFrame:
        """ Download and return market data """
        pass