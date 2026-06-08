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

    
    def generate_schedule(self):
        """ Payment schedule generator for swaps and other IR instruments """
        # check if aged_schedule exists
        if hasattr(self, '_aged_schedule') and self._aged_schedule is not None:
            return self._aged_schedule.tolist()
        
        # time between two consecutive payments
        step = 1.0 / self.periods_per_year

        schedule = np.arange(
            step,
            self.maturity + step,
            step
        )

        return schedule.tolist()
    

    def aged_schedule(
            self,
            valuation_time: float
    ):
        """ Returns remaining cashflow schedule after valuation_time """
        full_schedule = np.array(self.generate_schedule())

        remaining_schedule = full_schedule[full_schedule > valuation_time]

        # adjusted schedule by shifting to time o perspective
        adjusted_schedule = remaining_schedule - valuation_time

        new_schedule = PaymentSchedule(
            maturity = max(self.maturity - valuation_time, 0),
            frequency = self.freq
        )

        # override to aged schedule
        self._aged_schedule = None
        new_schedule._aged_schedule = adjusted_schedule

        return new_schedule