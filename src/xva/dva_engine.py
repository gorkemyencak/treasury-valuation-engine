import pandas as pd

from src.curves.discount_curve import DiscountCurve
from src.curves.survival_curve import SurvivalCurve

from src.risk.exposure.stochastic_exposure_engine import MonteCarloExposureEngine

class DVAEngine:
    """ Debit Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine,
            discount_curve: DiscountCurve,
            survival_curve: SurvivalCurve,
            recovery_rate: float = 0.4
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.survival_curve = survival_curve
        self.recovery_rate = recovery_rate

    def dva_profile(
            self,
            swap,
            n_paths = 250,
            steps_per_year: int = 4
    ):
        """
        DVA contribution by time bucket

        DVA Formula:
            DVA = (1 - R) sum_{i=1,..,n} [ENE(t_{i}) DF(t_{i}) PD(t_{i-1}, t_{i})]        
        """
        # expected negative exposure
        ene_profile = self.exposure_engine.expected_negative_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        rows = []

        previous_time = 0.0

        dva = 0.0

        for _, row in ene_profile.iterrows():

            current_time = row['Times']
            ENE = row['ENE']

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            PD = self.survival_curve.default_probability(
                start = previous_time,
                end = current_time
            )

            dva_contribution = (
                (1 - self.recovery_rate)
                * ENE
                * DF
                * PD
            )

            dva += dva_contribution

            rows.append({
                'Times': current_time,
                'ENE': ENE,
                'DF': DF,
                'PD': PD,
                'DVA_Contribution': dva_contribution
            })

            previous_time = current_time
        
        return pd.DataFrame(rows), dva