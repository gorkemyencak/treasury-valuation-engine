import numpy as np
import pandas as pd

from src.risk.collateral.csa import CSAAgreement

class NettingSet:
    """
    Portfolio netting set under a CSA agreement

    Aggregates trade-level PVs into a netted exposure profile and applies collateral mechanics    
    """
    def __init__(
            self,
            trades: list,
            csa_agreement: CSAAgreement | None = None,
            netting_id: str | None = None
    ):
        # attributes
        self.trades = trades
        self.csa_agreement = csa_agreement
        self.netting_id = netting_id

    @property
    def n_trades(self):
        return len(self.trades)
    
    def add_trade(
            self,
            trade
    ):
        """ Add trade to the netting set """
        self.trades.append(trade)

    def remove_trade(
            self,
            idx: int
    ):
        """ Remove a trade from netting set by index """
        del self.trades[idx]

    def aggregate_pv_matrix(
            self,
            pv_matrices: list[np.ndarray]
    ):
        """
        Aggregate trade PV matrices into portfolio PV matrix

        Formula:
            V_{net}(t, w) = sum_{i=1,..,n} V_{i}(t, w)
        """
        if len(pv_matrices) != self.n_trades:
            raise ValueError('Number of trades must be equal to the number of pv_matrices!')
        
        net_pv = np.sum(pv_matrices, axis = 0)

        return net_pv
    
    def collateralized_exposure(
            self,
            net_pv_matrix: np.ndarray
    ):
        """
        Apply CSA collateral to netted PV matrix

        Formula:
            E_{t}(w) = max(V_{net} - C_{t}, 0)
        """
        if self.csa_agreement is None:
            return np.maximum(net_pv_matrix, 0)
        
        collateral_matrix = np.zeros_like(net_pv_matrix)

        for i in range(net_pv_matrix.shape[0]):

            for j in range(net_pv_matrix.shape[1]):

                collateral_matrix[i, j] = self.csa_agreement.required_collateral(portfolio_value = net_pv_matrix[i, j])

        exposure_matrix = np.maximum(net_pv_matrix - collateral_matrix, 0)

        return exposure_matrix   
    
    def netting_benefit(
            self,
            pv_matrices: list[np.ndarray]
    ):
        """
        Returns netting benefit

        Formula:
            Netting_benefit = sum_{i=1,..,n} max(V_{i}, 0) - max(sum_{i=1,..,n} V_{i}, 0)        
        """
        # exposure w/out netting
        exposure_wout_netting = np.sum(
            [np.maximum(pv, 0) for pv in pv_matrices],
            axis = 0
        )

        # exposure with netting
        exposure_with_netting = np.maximum(
            self.aggregate_pv_matrix(pv_matrices = pv_matrices),
            0
        )
        
        # compute netting benefit
        benefit = exposure_wout_netting - exposure_with_netting

        return benefit
    
    def summary(self):
        ''' Summary table representing the trades benefiting from netting '''
        return pd.DataFrame({
            'NettingID': [self.netting_id],
            'Trades': [self.n_trades],
            'CSAEnabled': [self.csa_agreement is not None]
        })