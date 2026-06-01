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
        """ 
        Simple compounding discount factor approximation 
        
        DF Bootstrap Formula:
            DF(T_{2}) = DF(T_{1}) / (1 + F(T_{1}, T_{2})Δt)
        """
        # convert market quote into decimal points
        rate = self.market_rate / 100.0

        if curve is None or len(curve) == 0:
            # continuous DF approximation
            df = 1.0 / (1.0 + rate * self.maturity)

            return df
        
        previous_maturity = max(curve.keys())

        previous_df = curve[previous_maturity]

        delta_t = self.maturity - previous_maturity

        df = previous_df / (1.0 + rate * delta_t)
        
        return df