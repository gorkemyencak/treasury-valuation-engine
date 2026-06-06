import numpy as np
import pandas as pd

from src.portfolio.portfolio import Portfolio

from src.risk.ir_risk_engine import IRRiskEngine

class PortfolioRiskEngine:
    """ Portfolio risk analytics """
    def __init__(
            self,
            ir_risk_engine: IRRiskEngine
    ):
        # attributes
        self.ir_risk_engine = ir_risk_engine

    # risk analytics layer
    def portfolio_pv(
            self,
            portfolio
    ):
        """ 
        Return portfolio-level present value 
        
        Formula:
            PV_{portfolio} = sum_{i=1,..,n} (PV_{i})
        """
        return sum(
            self.ir_risk_engine.pricer.price(swap = trade)
            for trade in portfolio.trades
        )
    
    def portfolio_pv01(
            self,
            portfolio,
            shock_in_bps: int = 1
    ):
        """  
        Return portfolio-level PV01

        Formula:
            PV01_{portfolio} = sum_{i=1,..,n} (PV01_{i})        
        """
        return sum(
            self.ir_risk_engine.pv01(
                swap = trade,
                shock_in_bps = shock_in_bps
            )
            for trade in portfolio.trades
        )
    
    def portfolio_dv01(
            self,
            portfolio,
            shock_in_bps: int = 1
    ):
        """ 
        Return portfolio-level DV01 
        
        Formula:
            DV01_{portfolio} = sum_{i=1,..,n} (DV01_{i})
        """
        return sum(
            self.ir_risk_engine.dv01(
                swap = trade,
                shock_in_bps = shock_in_bps
            )
            for trade in portfolio.trades
        )
    
    def portfolio_convexity(
            self,
            portfolio,
            shock_in_bps: int = 1
    ):
        """  
        Return portfolio-level convexity

        Formula:
            Convexity_{portfolio} = sum{i=1,..,n} (Convexity_{i})        
        """
        return sum(
            self.ir_risk_engine.convexity(
                swap = trade,
                shock_in_bps = shock_in_bps
            )
            for trade in portfolio.trades
        )
    
    def portfolio_key_rate_dv01(
            self,
            portfolio,
            maturity: float,
            shock_in_bps: int = 1
    ):
        """ Return portfolo-level key-rate DV01 """
        return sum(
            self.ir_risk_engine.key_rate_dv01(
                swap = trade,
                maturity = maturity,
                shock_in_bps = shock_in_bps
            )
            for trade in portfolio.trades
        )
    

    # reporting layer
    def portfolio_risk_report(
            self,
            portfolio,
            shock_in_bps: int = 1
    ) -> pd.DataFrame:
        """ Return IR risk metrics of a swap portfolio """
        return pd.DataFrame({
            'Metric': [
                'Portfolio PV',
                'Portfolio PV01',
                'Portfolio DV01',
                'Portfolio Convexity'
            ],
            'Value': [
                self.portfolio_pv(portfolio = portfolio),
                self.portfolio_pv01(portfolio = portfolio, shock_in_bps = shock_in_bps),
                self.portfolio_dv01(portfolio = portfolio, shock_in_bps = shock_in_bps),
                self.portfolio_convexity(portfolio = portfolio, shock_in_bps = shock_in_bps)
            ]
        })
    
    def portfolio_key_rate_report(
            self,
            portfolio,
            shock_in_bps: int = 1
    ):
        """ Return key-rate DV01 of a swap portfolio for all key-rate tenors """
        key_rates = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        row_metrics = []

        for maturity in key_rates:

            krdv01_total = self.portfolio_key_rate_dv01(
                portfolio = portfolio,
                maturity = maturity,
                shock_in_bps = shock_in_bps
            )                

            row_metrics.append({
                'Maturity': maturity,
                'KRDV01': krdv01_total
            })

        return pd.DataFrame(row_metrics)
    

    


    



