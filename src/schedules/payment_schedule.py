import numpy as np

class PaymentSchedule:
    """ Create payment schedule for swaps and other IR instruments """
    FREQUENCY_MAP = {
        'monthly': 12,
        'quarterly': 4,
        'semiannual': 2,
        'annual': 1
    }

    def __init__(
            self,
            maturity: float,
            frequency: str
    ):
        # attributes
        self.maturity = maturity
        self.freq = frequency

        self.periods_per_year = self.FREQUENCY_MAP[self.freq]

        self._aged_dates = None
        self._original_dates = None

    
    def generate_schedule(self):
        """ Payment schedule for swaps and other IR instruments measured from current validation date """
        if self._aged_dates is not None:
            return self._aged_dates.tolist()

        # time between two consecutive payments
        step = 1.0 / self.periods_per_year

        schedule = np.arange(
            step,
            self.maturity + step,
            step
        )

        return schedule.tolist()
    

    def generate_original_schedule(self):
        """ Original payment dates measured from trade inception """
        if self._original_dates is not None:
            return self._original_dates.tolist()
        
        return self.generate_schedule()
    

    def aged_schedule(
            self,
            valuation_time: float
    ):
        """  
        Remaining schedule after valuation_time

        e.g.
        Original:
            -> [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        valuation_time = 1.25

        Original remaining:
            -> [1.5, 2.0, 2.5, 3.0]
        
        Aged dates:
            -> [0.25, 0.75, 1.25, 1.75]
        """
        full_schedule = np.array(self.generate_schedule())

        remaining_schedule = full_schedule[full_schedule > valuation_time]

        aged_dates = remaining_schedule - valuation_time

        new_schedule = PaymentSchedule(
            maturity = max(self.maturity - valuation_time, 0.0),
            frequency = self.freq
        )

        new_schedule._aged_dates = aged_dates
        new_schedule._original_dates = remaining_schedule

        return new_schedule