import pandas as pd

from src.curves.discount_curve import DiscountCurve

from src.risk.exposure.stochastic_exposure_engine_2factor import MonteCarloExposureEngine2Factor

class FVAEngine:
    """ Funding Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine2Factor,
            discount_curve: DiscountCurve,
            funding_spread
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.funding_spread = funding_spread

    def fva_profile(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ 
        FVA contribution by time bucket 
        
        FVA Formula:
            FVA = s_{f} sum_{i=1,..,n} [EE(t_{i}) DF(t_{i}) Δt_{i}]
        """
        # expected exposure profile
        ee_profile = self.exposure_engine.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        rows = []

        previous_time = 0.0
        fva = 0.0

        for _, row in ee_profile.iterrows():

            current_time = row['Times']
            EE = row['EE']

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            dt = current_time - previous_time

            fva_contribution = (
                self.funding_spread
                * EE
                * DF
                * dt
            )

            fva += fva_contribution

            rows.append({
                'Times': current_time,
                'EE': EE,
                'DF': DF,
                'dt': dt,
                'FVA_Contribution': fva_contribution
            })

            previous_time = current_time
        
        return pd.DataFrame(rows), fva