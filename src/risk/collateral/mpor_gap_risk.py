import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

from src.risk.collateral.netting_set import NettingSet
from src.risk.exposure.stochastic_exposure_engine_2factor import MonteCarloExposureEngine2Factor

class MPORGapRiskEngine:
    """
    Margin Period of Risk (MPOR) gap risk engine

    MPORGapRiskEngine measures exposure during collateral freeze window after default    
    """
    def __init__(
            self,
            exposure_engine: MonteCarloExposureEngine2Factor,
            netting_engine: NettingSet
    ):
        # attributes
        self.exposure_engine = exposure_engine
        self.netting_set = netting_engine

    # helper function
    def _portfolio_maturity(self):
        """ Return the maximum maturity of trades within the portfolio """
        # portfolio maturity
        max_maturity = max(
            trade.maturity
            for trade in self.netting_set.trades
        )

        return max_maturity

    # risk analytics layer
    def gap_exposure_matrix(
            self,
            n_paths: int = 250,
            steps_per_year: int = 252
    ):
        """
        Return gap exposure matrix during the margin period of risk (Δ days)
        
        Gap exposure:
            G_{t} = max(V_{t + Δ} - C_{t}, 0)
        
        """
        # trade-level PV matrices
        pv_matrices = []

        for trade in self.netting_set.trades:

            trade_pv = self.exposure_engine.simulate_pv_matrix(
                swap = trade,
                n_paths = n_paths,
                steps_per_year = steps_per_year
            )

            pv_matrices.append(trade_pv)

        # aggregate net PV
        net_pv = self.netting_set.aggregate_pv_matrix(
            pv_matrices = pv_matrices
        )

        # MPOR shift in grid steps
        if self.netting_set.csa_agreement is None:
            raise ValueError('CSA Agreement is required!')
        
        mpor_days = self.netting_set.csa_agreement.mpor

        mpor_time_steps = max(
            1,
            int((mpor_days / 252) * steps_per_year)
        )

        _n_paths, n_times = net_pv.shape

        gap_matrix = np.zeros_like(net_pv)

        for i in range(_n_paths):

            for t in range(n_times):

                future_idx = min(
                    t + mpor_time_steps,
                    n_times - 1
                )

                # collateral frozen at t
                collateral_t = self.netting_set.csa_agreement.total_collateral_held(
                    portfolio_value = net_pv[i, t]
                )

                # closeout value at t + Δ
                future_value = net_pv[i, future_idx]

                # gap exposure
                gap_matrix[i, t] = max(
                    future_value - collateral_t,
                    0
                )

        return gap_matrix
    
    def expected_gap_exposure(
            self,
            n_paths: int = 250,
            steps_per_year: int = 252
    ):
        """ 
        Return expected MPOR gap exposure 
        
        Formula:
            EE_{t}(MPOR) = E[max(V_{t + Δ} - C_{t}, 0)]
        """
        # retrieve gap matrix
        gap_matrix = self.gap_exposure_matrix(
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # longest maturity cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = self._portfolio_maturity(),
            steps_per_year = steps_per_year
        )

        # expected gap exposure
        ee_gap = gap_matrix.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'EE_MPOR': ee_gap
        })
    
    def potential_future_gap_exposure(
            self,
            percentile: float = 95.0,
            n_paths: int = 250,
            steps_per_year: int = 252
    ):
        """
        Return potential future MPOR gap exposure

        Formula:
            PFE_{t}(MPOR) = Q_{q}(G_{t})
        """
        # retrieve gap matrix
        gap_matrix = self.gap_exposure_matrix(
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # longest maturity cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = self._portfolio_maturity(),
            steps_per_year = steps_per_year
        )

        # potential future gap exposure
        q = percentile / 100.0

        pfe_gap = np.quantile(
            gap_matrix,
            q,
            axis = 0
        )

        return pd.DataFrame({
            'Times': cf_grid,
            f'PFE_{percentile:.0f}%': pfe_gap
        })