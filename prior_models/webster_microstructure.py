import sys
import pandas as pd
from prior_models.prior_model import Prior

PATH = "../../../taq_db/TAQ-Query-Scripts/data/raw_data/2020-01-02/"
sys.path.append(PATH)
DATA_FILE = "AMZN_trades_cleaned.csv"


class Webster_MicroStructure(Prior):

    # ref: 

    def __init__(self, n, time_flag=False, t=None):

        self.name = "Webster_MicroStructure"

        market_data = pd.read_csv(PATH + DATA_FILE, index_col=0, nrows=2 * n)

        # fetch the last n volume, spread, etc.

        volume_data = market_data["Trade_Volume"][market_data["Trade_Volume"] < 1000][-n:]

        spread_data = market_data["Effective_Spread"][-n:]

        prior=None

        # fetch time series if time_flag is true
        if time_flag:
            volume_data.index = pd.to_datetime(volume_data.index)
            t = [(i - volume_data.index[0]).total_seconds() for i in volume_data.index]

            # update time_flag and t
            self.time_flag = time_flag
            self.t = t
        
        # update the prior
        super().__init__(prior, t)
        self.name = "Webster_MicroStructure"
        self.prior = prior



    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):

        return self.time_flag