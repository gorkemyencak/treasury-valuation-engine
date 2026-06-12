import numpy as np
import pandas as pd

class PCAFactorExtractor:
    """ 
    PCA factor extractor for historical yield curves
    
    Extracting level and slope factors (x_t, y_t) from historical daily yield changes using PCA 

    Returns historical factor series that can be used for HW 2-Factor calibration
    """
    def __init__(
            self,
            n_factors: int = 2
    ):
        # attributes
        self.n_factors = n_factors

        self.eigenvalues = None
        self.eigenvectors = None
        self.factor_loadings = None
        self.explained_variance_ratio = None

    def fit(
            self,
            yield_curve_history: pd.DataFrame
    ):
        """  
        Fit PCA to historical yield curve changes

        Parameter:
            - yield_curve_history
                row -> dates
                column -> tenors

        Eigen Decomposition:
            Σ q_{i} = lambda_{i} * q_{i}
            
            where
                - Σ: covariance matrix -> Σ = Cov(X) 
                    -> X: [Δy_{1},.., Δy_{n}]
                - q_{i}: eigenvector
                - lambda_{i}: eigenvalue

        Explained Variance:
            EV_{i} = lambda_{i} / sum_{j=1,..,n} lambda_{j}
        """
        # yield curve changes -> (dates x tenors)
        yield_change = yield_curve_history.diff().dropna()

        # covariance matrix -> (tenors x tenors)
        X = yield_change.values
        cov_matrix = np.cov(X.T)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # descending order
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        self.eigenvalues = eigenvalues

        eigenvectors = eigenvectors[:, idx]
        self.eigenvectors = eigenvectors

        self.explained_variance_ratio = eigenvalues / eigenvalues.sum()

        return self
    

    def transform(
            self,
            yield_curve_history: pd.DataFrame
    ):
        """ 
        Projects yield changes onto PCA directions and reconstructs cumulative factor levels

        Remark:
            PCA is fitted on Δy_{t} (yield changes), so the raw projection gives factor shocks Δf_{t}

            HW2F OU calibration requires factor levels f_{t}, so we require to sum the shocks cumulatively!
        """
        if self.eigenvectors is None:
            raise ValueError('Fit PCAFactorExtractor first!')
        
        # daily yield changes -> (dates x tenors)
        yield_change = yield_curve_history.diff().dropna()
        X = yield_change.values

        # PCA-projected factor shocks -> (dates x n_factors)
        factors = X @ self.eigenvectors[:, :self.n_factors]

        # convert projected factor shocks into factor levels
        factor_levels = np.cumsum(factors, axis = 0)

        return pd.DataFrame(
            factor_levels,
            index = yield_change.index,
            columns = [
                f'Factor_{i+1}'
                for i in range(self.n_factors)
            ]
        )
    
    def fit_transform(
            self,
            yield_curve_history: pd.DataFrame
    ):
        """ Fit PCA and return factor series """
        # fit PCA
        self.fit(yield_curve_history = yield_curve_history)

        return self.transform(yield_curve_history = yield_curve_history)
    
    def variance_report(self):
        """ Explained variance report """
        if self.explained_variance_ratio is None:
            raise ValueError('Fit PCAFactorExtractor first!')
        
        return pd.DataFrame({
            'Component': [
                f'PC{i+1}'
                for i in range(len(self.explained_variance_ratio))
            ],
            'ExplainedVariance': self.explained_variance_ratio,
            'CumulativeVariance': np.cumsum(self.explained_variance_ratio)
        })
    
    def get_factor_loadings(self):
        """ Eigenvector loadings """
        if self.eigenvectors is None:
            raise ValueError('Fit PCAFactorExtractor first!')
        
        self.factor_loadings = self.eigenvectors[:, :self.n_factors]
        
        return pd.DataFrame(
            self.factor_loadings,
            columns = [
                f'Factor{i+1}'
                for i in range(self.n_factors)
            ]
        )