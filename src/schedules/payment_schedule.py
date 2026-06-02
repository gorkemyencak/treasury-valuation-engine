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
        # time between two consecutive payments
        step = 1.0 / self.periods_per_year

        schedule = np.arange(
            step,
            self.maturity + step,
            step
        )

        return schedule.tolist()