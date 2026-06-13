
class CSAAgreement:
    """  
    Credit Support Annex (CSA) container

    CSA container defines collateral mechanics such as:
        - threshold (H)
        - minimum transfer amount (MTA)
        - independent amount (IA)
        - margin period of risk (MPOR)    

    Portfolio Value:
        V_{net}(t) = sum_{i=1,..,n}(V_{i}(t))

    Expected Exposure:
        EE_{net}(t) = E[max(V_{net}(t), 0)]
    """
    def __init__(
            self,
            threshold: float = 0.0,
            minimum_transfer_amount: float = 0.0,
            independent_amount: float = 0.0,
            margin_period_of_risk: int = 10
    ):
        # attributes
        self.threshold = threshold
        self.mta = minimum_transfer_amount
        self.ia = independent_amount
        self.mpor = margin_period_of_risk

    def required_collateral(
            self,
            portfolio_value: float
    ) -> float:
        """  
        Variation margin required under CSA

        Formula:
            VM_{t} = max(V_{t} - H, 0)

            where
                - H: threshold        
        """

        return max(portfolio_value - self.threshold, 0)
    
    def collateral_call(
            self,
            portfolio_value: float,
            current_collateral: float = 0.0
    ) -> float:
        """  
        Increment collateral call

        Formula:
            Call_{t} = C_{t}(required) - C_{t}(current)

            where 
                - C_{t}(required): required collateral to be posted at time t
                - C_{t}(current): collateral posted at time t

        If |Call_{t}| < MTA:
            Call_{t} = 0
        else:
            Call_{t} > 0
        """
        # required collateral
        required_call = self.required_collateral(portfolio_value = portfolio_value)

        # incremental collateral call
        call_amount = required_call - current_collateral

        if abs(call_amount) < self.mta:
            return 0.0
        
        return call_amount
    
    def total_collateral_held(
            self,
            portfolio_value: float
    ) -> float:
        """
        Total collateral held

        Formlua:
            C_{t} = VM_{t} + IA

            where
                - VM_{t}: variation margin at time t
                - IA: independent amount -> an extra collateral posted upfront
        """
        # variation margin
        variation_margin = self.required_collateral(portfolio_value = portfolio_value)

        return variation_margin + self.ia
    
    def collateralized_exposure(
            self,
            portfolio_value: float
    ) -> float:
        """
        Residual collateralized exposure

        Formula:
            E_{t} = max(V_{t} - C_{t}, 0)
        """
        # collateral held
        collateral = self.total_collateral_held(portfolio_value = portfolio_value)

        # residual exposure
        residual_exposure = max(portfolio_value - collateral, 0)

        return residual_exposure
    
    def summary(self):
        """ Summary dictionary representing CSA container attributes """
        return{
            'Threshold': self.threshold,
            'MTA': self.mta,
            'IndependentAmount': self.ia,
            'MPOR_Days': self.mpor
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"Threshold={self.threshold}, "
            f"MTA={self.mta}, "
            f"IA={self.ia}, "
            f"MPOR={self.mpor})"
        )