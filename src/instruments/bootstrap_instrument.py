from abc import ABC, abstractmethod

class BootstrapInstrument(ABC):
    """ Abstract bootstrap instruments that all other curve instruments inherit from this class """
    def __init__(
            self,
            instrument_type: str,
            tenor: str,
            market_rate: float
    ):
        # attributes
        self.instrument_type = instrument_type
        self.tenor = tenor
        self.market_rate = market_rate

    @abstractmethod
    def implied_discount_factor(
        self,
        curve = None
    ) -> float:
        """ Compute implied discount factor from market quote """
        raise NotImplementedError

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'type={self.instrument_type}, '
            f'tenor={self.tenor}, '
            f'rate={self.market_rate}'
            f')'
        )