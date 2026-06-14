import numpy as np
import pandas as pd

from src.utils.cashflow_grid import CashflowGridder

from src.risk.exposure.stochastic_exposure_engine_2factor import MonteCarloExposureEngine2Factor

from src.risk.collateral.netting_set import NettingSet

class CollateralizedMonteCarloExposureEngine:
    """
    Collateralized Monte Carlo exposure engine

    Portfolio-level exposure engine with:
        - Trade-level HW2F Monte Carlo valuation
        - Netting aggregation
        - CSA collateralization mechanics    
    """
    def __init__(
            self,
            trade_exposure_engine: MonteCarloExposureEngine2Factor,
            netting_engine: NettingSet
    ):
        # attributes
        self.trade_engine = trade_exposure_engine
        self.netting = netting_engine

    # helper function
    def _portfolio_maturity(self):
        """ Return the maximum maturity of trades within the portfolio """
        # portfolio maturity
        max_maturity = max(
            trade.maturity
            for trade in self.netting.trades
        )

        return max_maturity

    # risk analytics layer
    def collateralized_exposure_matrix(
            self,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return full collateralized portfolio exposure matrix

        Steps to follow:
            1- Trade PVs
            2- Netting
            3- CSA Collateral
            4- Residual Exposure      
        """
        ### step1: simulate trade-level PV matrices
        # initialize pv_matrices -> (n_trades x n_paths x n_times)
        pv_matrices = []

        for trade in self.netting.trades:

            trade_pv = self.trade_engine.simulate_pv_matrix(
                swap = trade,
                n_paths = n_paths,
                steps_per_year = steps_per_year
            )

            pv_matrices.append(trade_pv)
        
        ### step2: aggregate portfolio netting
        # aggregate trade PV matrices into portfolio PV matrix
        net_pv_matrix = self.netting.aggregate_pv_matrix(
            pv_matrices = pv_matrices
        )

        ### step 3: apply CSA collateral
        # applying CSA collateral to netted PV matrix and return residual exposure
        collateralized_pv_matrix = self.netting.collateralized_exposure(
            net_pv_matrix = net_pv_matrix
        )

        return collateralized_pv_matrix
    
    def expected_exposure(
            self,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return collateralized expected exposure profile

        Formula:
            EE_{t} = E[E_{t}(w)]
        """
        # exposure matrix
        coll_exposure_matrix = self.collateralized_exposure_matrix(
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = self._portfolio_maturity(),
            steps_per_year = steps_per_year
        )

        # collateralized expected exposure
        ee_coll = coll_exposure_matrix.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'CollateralizedEE': ee_coll
        })
    
    def potential_future_exposure(
            self,
            percentile: float = 95.0,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Return collateralized PFE profile

        Formula:
            PFE_{t} = Quantile_{q}(E_{t}(w))
        """
        # exposure matrix
        coll_exposure_matrix = self.collateralized_exposure_matrix(
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        # cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = self._portfolio_maturity(),
            steps_per_year = steps_per_year
        )

        # collateralized potential future exposure
        pfe_coll = np.percentile(
            coll_exposure_matrix,
            percentile,
            axis = 0
        )

        return pd.DataFrame({
            'Times': cf_grid,
            f'PFE_{percentile:.0f}%': pfe_coll
        })
    
    def netting_benefit(
            self,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """
        Compute portfolio-level netting benefit

        Formula:
            Netting_benefit = sum_{i=1,..,n} max(V_{i}, 0) - max(sum_{i=1,..,n} V_{i}, 0)  
        """
        # initialize pv_matrices -> (n_trades x n_paths x n_times)
        pv_matrices = []

        for trade in self.netting.trades:

            trade_pv = self.trade_engine.simulate_pv_matrix(
                swap = trade,
                n_paths = n_paths,
                steps_per_year = steps_per_year
            )

            pv_matrices.append(trade_pv)
        
        # cf_grid
        cf_grid = CashflowGridder.cf_grid(
            maturity = self._portfolio_maturity(),
            steps_per_year = steps_per_year
        )

        # return netting benefit matrix
        benefit_matrix = self.netting.netting_benefit(
            pv_matrices = pv_matrices
        )

        # compute portfolio-level netting benefit
        benefit = benefit_matrix.mean(axis = 0)

        return pd.DataFrame({
            'Times': cf_grid,
            'NettingBenefit': benefit
        })