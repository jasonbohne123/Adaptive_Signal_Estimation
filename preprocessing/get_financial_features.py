import sys

sys.path.append("../../../taq_db/TAQ-Query-Scripts/data/features/2020-01-02/")

import numpy as np
import pandas as pd


def get_financial_features(
    path="../../../taq_db/TAQ-Query-Scripts/data/features/2020-01-02/",
    file_name="GOOG_reconstructed_features.csv",
    dtype="trades",
    n=1000,
):
    """Helper Function to extract features of interest into estimation library"""

    features = pd.read_csv(path + file_name, index_col=0, low_memory=False)

    # separate trade and quote features
    trade_features = features[features["Trade_Price"].notnull()]
    quote_features = features[(features["Bid_Price"].notnull()) & (features["Offer_Price"].notnull())]

    # drop columns with all null values
    trade_features = trade_features.dropna(axis=1, how="all")
    quote_features = quote_features.dropna(axis=1, how="all")

    # select relevant columns and filter conditions
    trade_features = trade_features[trade_features["Sale_Condition"] == "@   "]
    trade_features = trade_features[["Trade_Volume", "Trade_Price"]]

    quote_features = quote_features[quote_features["National_BBO_Indicator"] == 4]
    quote_features = quote_features[
        ["Midprice", "Bid_Size", "Offer_Size", "Imbalance", "Effective_Spread", "Microprice"]
    ]

    if dtype == "trades":
        return trade_features[:n]
    elif dtype == "quotes":
        return quote_features[:n]


if __name__ == "__main__":
    print(get_financial_features(dtype="quotes").head())
