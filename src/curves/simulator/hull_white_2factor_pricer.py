import numpy as np

from src.curves.zero_curve import ZeroCurve

class HullWhite2FactorPricer:
    """ 
    Gaussian 2-factor Hull-White bond pricer engine 

    HullWhite2FactorPricer prices the instruments using generated factor paths without needing a full yield curve
    
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

        self.discount_curve = self.zero_curve.to_discount_curve()

        # caches
        self.df_cache = {}
        self.b1_cache = {}
        self.b2_cache = {}
        self.var_cache = {}

    # helper function
    def _cached_df(
            self,
            maturity: float
    ):
        """ Helper to store discount factor in the cache to speed up the total runtime """
        if maturity not in self.df_cache:
            self.df_cache[maturity] = self.discount_curve.get_discount_factor(maturity = maturity)

        return self.df_cache[maturity]    
    
    # B terms
    def B1(self, t, T):
        #return (1 - np.exp(-self.a * (T - t))) / self.a
        key = (t, T)

        if key not in self.b1_cache:
            
            self.b1_cache[key] = (1 - np.exp(-self.a * (T - t))) / self.a
        
        return self.b1_cache[key]
    
    def B2(self, t, T):
        #return (1 - np.exp(-self.b * (T - t))) / self.b
        key = (t, T)

        if key not in self.b2_cache:
            
            self.b2_cache[key] = (1 - np.exp(-self.b * (T - t))) / self.b
        
        return self.b2_cache[key]
    
    # variance term
    '''
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

        return V1 + V2 + V12 '''
    def V(self, t, T):

        key = (t, T)

        if key not in self.var_cache:

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

            self.var_cache[key] = V1 + V2 + V12
        
        return self.var_cache[key]
    
    def bond_price(
            self,
            t,
            T,
            x_t,
            y_t
    ):
        """ 
        Zero-coupon bond pricing function with 2-additive-factor Hull-White parameters 
            -> DF(t, T) = P(t, T)
        """
        # discount factors
        P_0T = self._cached_df(maturity = T) #self.discount_curve.get_discount_factor(maturity = T)
        P_0t = self._cached_df(maturity = t) #self.discount_curve.get_discount_factor(maturity = t)

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