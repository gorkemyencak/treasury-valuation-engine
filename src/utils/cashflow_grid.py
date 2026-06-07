import numpy as np

class CashflowGridder:


    @staticmethod
    def cf_grid(
            maturity: float,
            steps_per_year: int = 2       
    ) -> np.ndarray:
        """ Return cashflow grid of an instrument """
        return np.arange(
            0,
            maturity + 1/steps_per_year,
            1/steps_per_year
        )