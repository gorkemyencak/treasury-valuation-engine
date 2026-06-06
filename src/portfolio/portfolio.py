
class Portfolio:
    """ Portfolio container class  """
    def __init__(self):

        # attributes
        self.trades = []

    def add_trade(
        self,
        trade
    ):
        """ Incrementally append to trade list """
        self.trades.append(trade)
    
    def __len__(self):
        """ Return number of trades appended """
        return len(self.trades)
    
    def summary(self):
        return self.trades