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
        Continuously compounded discount factor approximation 
        
        OIS par swap condition: -> fixed-leg = floating-leg
            1 - DF(T_{n}) = R * sum_{i = 1,..,n}(alpha_{i} * DF(T_{i}))

            where
                R: market OIS rate
                alpha_{i}: accrual factor
                DF(T_{i}): discount factor at time i
        Let's assume alpha_{i} = 1.0 for 1Y OIS for simplification
        """
        # convert market quote into decimal points
        rate = self.market_rate / 100.0

        if curve is None or len(curve) == 0:
            # DF approximation
            df = np.exp(-rate * self.maturity)

            return df
        
        else:
            known_maturities = curve.keys()

            fixed_leg_pv = 0.0

            for maturity in known_maturities:

                if maturity < self.maturity:

                    df_i = curve[maturity]

                    fixed_leg_pv += df_i
                
            numerator = 1.0 - rate * fixed_leg_pv
            denominator = 1.0 + rate

            df = numerator / denominator

            return df