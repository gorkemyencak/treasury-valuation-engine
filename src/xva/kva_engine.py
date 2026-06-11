import pandas as pd

from src.curves.discount_curve import DiscountCurve

from src.risk.exposure.stochastic_exposure_engine_2factor import MonteCarloExposureEngine2Factor

class KVAEngine:
    """ Capital Valuation Adjustment Engine """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine2Factor,
            discount_curve: DiscountCurve,
            capital_ratio: float = 0.08,
            cost_of_capital: float = 0.10
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.discount_curve = discount_curve
        self.capital_ratio = capital_ratio
        self.cost_of_capital = cost_of_capital

    def kva_profile(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """  
        KVA contribution by time bucket

        KVA Formula:
            KVA = cost_of_capital * sum_{i=1,..,n} [Capital(t_{i}) DF(t_{i}) Δt_{i}]

            where
                - Capital(t_{i}) = capital_ratio * EE(t_{i})
        """
        # expected exposure profile
        ee_profile = self.exposure_engine.expected_exposure(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        rows = []

        previous_time = 0.0
        kva = 0.0

        for _, row in ee_profile.iterrows():

            current_time = row['Times']
            EE = row['EE']

            capital = self.capital_ratio * EE

            DF = self.discount_curve.get_discount_factor(maturity = current_time)

            dt = current_time - previous_time

            kva_contribution = (
                self.cost_of_capital
                * capital
                * DF
                * dt
            )

            kva += kva_contribution

            rows.append({
                'Times': current_time,
                'EE': EE,
                'Capital': capital,
                'DF': DF,
                'dt': dt,
                'KVA_Contribution': kva_contribution
            })

            previous_time = current_time

        return pd.DataFrame(rows), kva