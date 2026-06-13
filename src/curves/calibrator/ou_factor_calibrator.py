import numpy as np
import pandas as pd

class OUFactorCalibrator:
    """ 
    Calibrate OU (Ornstein-Uhlenbeck) process from historical factor series

    OUFactorCalibrator engine takes PCA factor history (x_t, y_t) and estimates the HW parameters:
        - mean reversion -> a, b
        - volatilities -> sigma1, sigma2
        - correlation -> rho

    For an emprical factor series extracted by PCA: (x_{0},...,x_{n})
        -> we fit 
            dx_{t} = -kappa x_{t} dt + sigma dW_{t}    -> Continuous OU-process

            where 
                kappa -> mean reversion parameters (a or b)

        -> AR(1) process
            x_{t+1} = phi x_{t} + epsilon_{t}

        -> revisiting exact discrete solution for continuous OU-process
            x_{t + Δt} = e^{-kappa Δt} x_{t} + epsilon_{t}

            where
                epsilon_{t} ~ N(0, (sigma_{epsilon})^{2})
            
            -> consider AR(1) process:
                phi = e^{-kappa Δt}

                -> mean reversion
                    kappa = - ln(phi) / Δt
                
                -> residual
                    epsilon_{t} = x_{t+1} - phi x_{t}                 

        -> Regression form:
            x_{t+1} = phi x_{t} + epsilon_{t}

            using OLS:
                -> phi = sum (x_{t} x_{t+1}) / sum((x_{t})^{2)

        -> OU exact variance (residual var)
            Var(epsilon) = (sigma^{2} / 2 * kappa) * (1 - e^{-2 * kappa * dt})

            -> solve for:
                sigma = (Var(epsilon) * 2 * kappa / (1 - e^{-2 * kappa * dt}))^{1/2}

        -> Correlation estimation
            corr(dW_{1}, dW_{2}) = rho

            -> empirical proxy:
                rho = corr(Δx_{t}, Δy_{t})
    """
    def __init__(
            self,
            dt: float = 1/252
    ):
        # attributes
        self.dt = dt

    def calibrate_factor(
            self,
            factor_series: pd.Series
    ):
        """ 
        Calibrate single OU factor:
            dX = -kappa X dt + sigma dW
        """
        # lagged observations
        x = factor_series.to_numpy(dtype = float)

        x_t = x[:-1]
        x_t1 = x[1:]

        # OLS regression
        phi, _ = np.polyfit(
            x_t.astype(np.float64), 
            x_t1.astype(np.float64), 
            deg = 1
        ) 
        phi = np.clip(phi, 1e-8, 0.999999)

        # mean reversion
        kappa = -np.log(phi) / self.dt

        # residual term
        epsilon = x_t1 - phi * x_t

        # residual variance
        epsilon_var = np.var(epsilon, ddof = 1)

        # converting residual variance into HW diffusion volatility
        sigma = np.sqrt(
            epsilon_var
            * (2 * kappa)
            / (1 - np.exp(-2 * kappa * self.dt))
        )

        return {
            'mean_reversion': float(kappa),
            'volatility': float(sigma),
            'phi': float(phi)
        }
    
    def calibrate_hw2f(
            self,
            factor_history: pd.DataFrame
    ):
        """ Calibrate HW 2-factor parameters from PCA factors """
        # PCA factor series (x_t, y_t)
        x_t = factor_history.iloc[:, 0]
        y_t = factor_history.iloc[:, 1]

        # calibrated HW mean reversion, volatility and slope parameters
        calibrated_params1 = self.calibrate_factor(factor_series = x_t)
        calibrated_params2 = self.calibrate_factor(factor_series = y_t)

        # correlation
        dx = x_t.diff().dropna()
        dy = y_t.diff().dropna()

        rho = np.corrcoef(dx, dy)[0, 1]

        return {
            'a': calibrated_params1['mean_reversion'],
            'b': calibrated_params2['mean_reversion'],
            'sigma1': calibrated_params1['volatility'],
            'sigma2': calibrated_params2['volatility'],
            'rho': float(rho)
        }