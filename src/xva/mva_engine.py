import pandas as pd

from src.curves.discount_curve import DiscountCurve

from src.risk.exposure.stochastic_exposure_engine_2factor import MonteCarloExposureEngine2Factor

class MVAEngine:
    """ Margin Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine2Factor,
            discount_curve: DiscountCurve,
            funding_spread: float,
            im_multiplier: float = 1.0,
            percentile: float = 95.0
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.funding_spread = funding_spread
        self.im_multiplier = im_multiplier
        self.percentile = percentile

    def mva_profile(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        MVA contribution by time bucket

        MVA Formula:
            MVA = s_{f} sum_{i=1,..,n} [IM(t_{i}) DF(t_{i}) Δt_{i}]

            where
                - IM(t_{i}): initial margin -> IM(t_{i}) = im_multiplier * PFE_{percentile}(t_{i})
        """
        # potential future exposure
        pfe_profile = self.exposure_engine.potential_future_exposure(
            swap = swap,
            percentile = self.percentile,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        pfe_col = f'PFE_{self.percentile:.0f}%'

        rows = []

        previous_time = 0.0
        mva = 0.0

        for _, row in pfe_profile.iterrows():

            current_time = row['Times']
            IM = row[pfe_col] * self.im_multiplier

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            dt = current_time - previous_time

            mva_contribution = (
                self.funding_spread
                * IM
                * DF
                * dt
            )

            mva += mva_contribution

            rows.append({
                'Times': current_time,
                'InitialMargin': IM,
                'DF': DF,
                'dt': dt,
                'MVA_Contribution': mva_contribution
            })

            previous_time = current_time
        
        return pd.DataFrame(rows), mva