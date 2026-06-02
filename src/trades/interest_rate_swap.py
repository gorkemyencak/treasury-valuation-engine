from src.schedules.payment_schedule import PaymentSchedule

class InterestRateSwap:
    """ Vanilla pay-fixed, receive-floating interest rate swap container """
    def __init__(
            self,
            notional: float,
            maturity: float,
            fixed_rate: float,
            pay_fixed: bool = True,
            fixed_freq: str = 'semiannual',
            floating_freq: str = 'quarterly',
            currency: str = 'USD'
    ):
        # attributes
        self.notional = notional
        self.maturity = maturity
        self.fixed_rate = fixed_rate
        self.pay_fixed = pay_fixed
        self.fixed_freq = fixed_freq
        self.float_freq = floating_freq
        self.currency = currency

        self.fixed_schedule = PaymentSchedule(
            maturity = self.maturity,
            frequency = self.fixed_freq
        )

        self.float_schedule = PaymentSchedule(
            maturity = self.maturity,
            frequency = self.float_freq
        )

    @property
    def direction(self):
        return 'PAY_FIXED' if self.pay_fixed else 'RECEIVE_FIXED'
    

    def summary(self):
        """ Summary table of an IRS """
        return {
            'Notional': self.notional,
            'Maturity': self.maturity,
            'FixedRate': self.fixed_rate,
            'Direction': self.direction,
            'Currency': self.currency,
            'FixedFrequency': self.fixed_freq,
            'FloatFrequency': self.float_freq
        }

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'Notional={self.notional},  '
            f'Maturity={self.maturity},  '
            f'FixedRate={self.fixed_rate},  '
            f'Direction={self.direction})'
        )