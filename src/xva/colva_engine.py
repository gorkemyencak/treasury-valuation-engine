import pandas as pd

from src.curves.discount_curve import DiscountCurve

from src.risk.exposure.stochastic_exposure_engine import MonteCarloExposureEngine

class ColVAEngine:
    """ Collateral Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine,
            discount_curve: DiscountCurve,
            funding_rate: float,
            collateral_rate: float
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.funding_rate = funding_rate
        self.collateral_rate = collateral_rate

        self.spread = self.funding_rate - self.collateral_rate

    def colva_profile(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ 
        ColVA contribution by time bucket 
        
        ColVA Formula:
            ColVA = (r_{f} - r_{c}) sum_{i=1,..,n} [EE(t_{i}) DF(t_{i}) Δt_{i}]

            where
                - r_{f}: funding rate
                - r_{c}: collateral remuneration rate
        """
        # expected exposure profile
        ee_profile = self.exposure_engine.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        rows = []

        previous_time = 0.0
        colva = 0.0

        for _, row in ee_profile.iterrows():

            current_time = row['Times']
            EE = row['EE']

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            dt = current_time - previous_time

            colva_contribution = (
                self.spread
                * EE
                * DF
                * dt
            )

            colva += colva_contribution

            rows.append({
                'Times': current_time,
                'EE': EE,
                'DF': DF,
                'dt': dt,
                'ColVA_Contribution': colva_contribution
            })

            previous_time = current_time

        return pd.DataFrame(rows), colva