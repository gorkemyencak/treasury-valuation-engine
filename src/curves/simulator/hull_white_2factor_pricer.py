import numpy as np
import pandas as pd

from src.curves.zero_curve import ZeroCurve

class HullWhite2FactorPricer:
    """ 
    Gaussian 2-factor Hull-White bond pricer engine 
    
    Bond Pricing Formula:
        P(t, T) = (P(0, T) / P(0, t)) * e^{-B_{1}(t, T) * x_{t} - B_{2}(t, T) * y_{t} + (1/2) * V(t, T)}

        where
            - B_{1}(t, T) = (1 - e^{-a * (T-t)}) / a
            - B_{2}(t, T) = (1 - e^{-b * (T-t)}) / b
            - V(t, T) = V_{1} + V_{2} + V_{12}
                where
                    - V_{1} = (sigma_{1})^{2} * ((1 - e^{-2a * t}) / 2a) * (B_{1})^{2}
                    - V_{2} = (sigma_{2})^{2} * ((1 - e^{-2b * t}) / 2b) * (B_{2})^{2}
                    - V_{12} = 2 * rho * sigma_{1} * sigma_{2} * ((1 - e^{-(a+b) * t}) / (a + b)) * B_{1} * B_{2}
    """
    def __init__(
            self,
            zero_curve: ZeroCurve,
            a: float = 0.10,
            b: float = 0.50,
            sigma1: float = 0.010,
            sigma2: float = 0.005,
            rho: float = 0.25
    ):
        # attribute
        self.zero_curve = zero_curve
        self.a = a
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
    
    # B terms
    def B1(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a
    
    def B2(self, t, T):
        return (1 - np.exp(-self.b * (T - t))) / self.b
    
    # variance term
    def V(self, t, T):

        # B terms
        B1 = self.B1(t = t, T = T)
        B2 = self.B2(t = t, T = T)

        # V terms
        V1 = (
            (self.sigma1 ** 2)
            * (1 - np.exp(-2 * self.a * t))
            / (2 * self.a)
            * (B1 ** 2)
        )

        V2 = (
            (self.sigma2 ** 2)
            * (1 - np.exp(-2 * self.b * t))
            / (2 * self.b)
            * (B2 ** 2)
        )

        V12 = (
            (2 * self.rho * self.sigma1 * self.sigma2)
            * (1 - np.exp(-(self.a + self.b) * t))
            / (self.a + self.b)
            * (B1 * B2)
        )

        return V1 + V2 + V12
    
    def bond_price(
            self,
            t,
            T,
            x_t,
            y_t
    ):
        """ Zero-coupon bond pricing function with 2-additive-factor Hull-White parameters """
        # P(t, T) = (P(0, T) / P(0, t)) * e^{-B_{1}(t, T) * x_{t} - B_{2}(t, T) * y_{t} + (1/2) * V(t, T)}
        # discount factors
        P_0T = self.zero_curve.to_discount_curve().get_discount_factor(maturity = T)
        P_0t = self.zero_curve.to_discount_curve().get_discount_factor(maturity = t)

        # B terms
        B1 = self.B1(t = t, T = T)
        B2 = self.B2(t = t, T = T)

        # variance
        variance = self.V(t = t, T = T)

        price = (
            (P_0T / P_0t)
            * np.exp(-B1 * x_t - B2 * y_t + (variance / 2))
        )

        return float(price)