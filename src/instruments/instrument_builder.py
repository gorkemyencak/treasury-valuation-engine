from src.curves.curve_snapshot import CurveSnapshot

from src.instruments.deposit_instrument import DepositInstrument
from src.instruments.ois_instrument import OISInstrument
from src.instruments.future_instrument import FutureInstrument

class InstrumentBuilder:
    """ Instrument builder from market snapshot objects """
    @staticmethod
    def build_deposit_instruments(snapshot: CurveSnapshot) -> list[DepositInstrument]:
        """ Deposit instrument builder from curve snapshot """
        instruments = []

        for tenor, rate in zip(snapshot.tenors, snapshot.rates):

            inst = DepositInstrument(
                tenor = tenor,
                market_rate = rate
            )

            instruments.append(inst)
        
        return instruments
    
    @staticmethod
    def build_future_instruments(snapshot: CurveSnapshot) -> list[FutureInstrument]:
        """ Future instrument builder from curve snapshot """
        instruments = []

        for tenor, rate in zip(snapshot.tenors, snapshot.rates):

            inst = FutureInstrument(
                tenor = tenor,
                market_rate = rate
            )

            instruments.append(inst)
        
        return instruments
    
    @staticmethod
    def build_ois_instruments(snapshot: CurveSnapshot) -> list[OISInstrument]:
        """ OIS instrument builder from curve snapshot """
        instruments = []

        for tenor, rate in zip(snapshot.tenors, snapshot.rates):

            inst = OISInstrument(
                tenor = tenor,
                market_rate = rate
            )

            instruments.append(inst)
        
        return instruments