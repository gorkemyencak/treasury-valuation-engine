import pandas as pd

from src.curves.discount_curve import DiscountCurve
from src.curves.survival_curve import SurvivalCurve

from src.risk.exposure.stochastic_exposure_engine import MonteCarloExposureEngine

class CVAEngine:
    """ Credit Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine,
            discount_curve: DiscountCurve,
            survival_curve: SurvivalCurve,
            recovery_rate: float = 0.40
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.survival_curve = survival_curve
        self.recovery_rate = recovery_rate

    def cva_profile(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ 
        CVA contribution by time bucket 
        
        CVA Formula:
            CVA = (1-R) sum_{i=1,..,n} [EE(t_{i}) DF(t_{i}) PD(t_{i-1}, t_{i})]
        """
        # expected exposure profile
        ee_profile = self.exposure_engine.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        rows = []

        previous_time = 0.0
        cva = 0.0

        for _, row in ee_profile.iterrows():

            current_time = row['Times']
            EE = row['EE']

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            PD = self.survival_curve.default_probability(
                start = previous_time,
                end = current_time
            )

            cva_contribution = (
                (1 - self.recovery_rate)
                * EE
                * DF
                * PD
            )

            cva += cva_contribution

            rows.append({
                'Times': current_time,
                'EE': EE,
                'DF': DF,
                'PD': PD,
                'CVA_Contribution': cva_contribution
            })

            previous_time = current_time
        
        return pd.DataFrame(rows), cva