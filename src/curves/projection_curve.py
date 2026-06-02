import pandas as pd

from src.curves.discount_curve import DiscountCurve

class ProjectionCurve():
    """ Forward rate projection curve """
    def __init__(
            self,
            discount_curve: DiscountCurve
    ):
        # attributes
        self.discount_curve = discount_curve
        
        self.maturities = self.discount_curve.maturities

        self.forward_rates = self._build_forward_curve()


    def _build_forward_curve(self):
        """
        Formula:
            F(t_{1}, t_{2}) = (1/alpha) * (DF(t_{1}) / DF(t_{2})) - 1

            where 
                alpha: year fraction
                DF(t_{1}): discount factor at the start
                DF(t_{2}): discount factor at the end
        """

        forwards = {}

        for i in range(len(self.maturities) - 1):
            
            # assign start & end times
            t1 = self.maturities[i]
            t2 = self.maturities[i+1]

            # assign discount factors for start & end times
            df1 = self.discount_curve.get_discount_factor(maturity = t1)
            df2 = self.discount_curve.get_discount_factor(maturity = t2)

            # year fraction
            alpha = t2 - t1

            # compute forward rate
            forward = (1 / alpha) * ((df1 / df2) - 1.0)

            forwards[(float(t1), float(t2))] = float(forward)

        return forwards
    

    def forward_rate(
            self,
            start,
            end
    ):
        """ Returning forward rate of a start & end times tuple """
        return self.forward_rates[(start, end)]
    
    def get_forward_rate(
            self,
            start,
            end
    ):
        """ Returning forward rate of an arbitrary (start, end) tuple """
        df1 = self.discount_curve.get_discount_factor(maturity = start)
        df2 = self.discount_curve.get_discount_factor(maturity = end)

        alpha = end - start

        return float((1 / alpha) * ((df1 / df2) - 1.0))
    

    def summary(self) -> pd.DataFrame:
        """ Return summary table consisting of forward rates of a given discount curve """
        return pd.DataFrame(
            [
                {
                    'Start': start,
                    'End': end,
                    'ForwardRate': rate * 100
                }
                for (start, end), rate in self.forward_rates.items()
            ]
        )



