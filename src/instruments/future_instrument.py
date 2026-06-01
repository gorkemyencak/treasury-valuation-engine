import numpy as np

from src.utils.parser import TenorParser

from src.instruments.bootstrap_instrument import BootstrapInstrument

class FutureInstrument(BootstrapInstrument):
    """ Future bootstrap instrument """
    def __init__(
            self, 
            tenor: str, 
            market_rate: float
    ):
        
        super().__init__(
            instrument_type = 'future', 
            tenor = tenor, 
            market_rate = market_rate
        )

        self.maturity = TenorParser.tenors_to_years(tenor = tenor)
    

    def implied_discount_factor(
            self, 
            curve = None
    ) -> float:
        """ Simple compounding discount factor approximation """
        # convert market quote into decimal points
        rate = self.market_rate / 100.0

        # continuous DF approximation
        df = 1.0 / (1.0 + rate * self.maturity)

        return df