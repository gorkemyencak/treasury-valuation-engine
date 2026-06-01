import numpy as np

from src.utils.parser import TenorParser

from src.instruments.bootstrap_instrument import BootstrapInstrument

class OISInstrument(BootstrapInstrument):
    """ OIS swap instrument """
    def __init__(
            self, 
            tenor: str, 
            market_rate: float
    ):
        
        super().__init__(
            instrument_type = 'ois_swap', 
            tenor = tenor, 
            market_rate = market_rate
        )

        self.maturity = TenorParser.tenors_to_years(tenor = tenor)

    
    def implied_discount_factor(
            self, 
            curve = None
    ) -> float:
        """ 
        Simple compounding discount factor approximation 
        
        -- Simplified version --
        Accrual recursive par-swap formula will be inteegrated in later stages of the project!!!
        """
        # convert market quote into decimal points
        rate = self.market_rate / 100.0

        # DF approximation
        df = 1.0 / (1.0 + rate * self.maturity)

        return df