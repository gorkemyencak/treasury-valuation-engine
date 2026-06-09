import numpy as np

class SurvivalCurve:
    """ Survival Curve denoting the probability that the issuer will not default over a specific time horizon """
    def __init__(
            self,
            hazard_rate: float
    ):
        
        # attributes
        self.hazard_rate = hazard_rate


    def survival_probability(
            self,
            t: float
    ):
        """ 
        Representing survival chance of the issuer given that the issuer has survived up to that point in time
        
        Formula:
            S_{t} = e^{-lambda * t)

            where 
                - lambda: hazard rate, the instantaneous rate of default at a specic moment        
        """
        return np.exp(-self.hazard_rate * t)
    

    def default_probability(
            self,
            start: float,
            end: float
    ):
        """ Return the default probability between two specific points in time """
        # compute PD
        probability_of_default = self.survival_probability(t = start) - self.survival_probability(t = end)

        return probability_of_default