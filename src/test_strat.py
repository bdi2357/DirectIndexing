import os,sys,re
import numpy as np
import pandas as pd
from dummy_strategy import dummy_wrapper
import time
import glob
from basic_stats import generate_basic_stats
if __name__ == "__main__":
    path_d = "../../DirectIndexingBondIt/data/holdings/IVV/"
    PriceVolume_dr = "../../DirectIndexingBondIt/data/PriceVolume/"
    index_df = pd.read_csv("../../DirectIndexingBondIt/data/index_data/SPY.csv")
    index_df = index_df.set_index("Date")
    close_col = [c for c in index_df.columns if c.lower().find("close") > -1]
    if len([c for c in close_col if c.lower().find("adj") > -1]) > 0:
        close_col = [c for c in close_col if c.lower().find("adj") > -1][0]
    else:
        close_col = close_col[0]
    index_df["return"] = getattr(index_df[close_col], "pct_change")()
    holdings_files = glob.glob(os.path.join(path_d, "*holdings_*.csv"))
    dts = [re.findall("20[0-9]+", x)[0] for x in holdings_files]
    dts.sort()
    match_d = {dts[ii]: dts[ii - 1] for ii in range(1, len(dts))}
    constraints = {}
    constraints["forbiden_tickers"] = ["MSFT","BAC","JPM","WFC","AXP"]
    start_dt = '2021-09-30'
    start = time.time()
    index_holdings_path = "../../DirectIndexingBondIt/data/holdings/IVV/"
    aprox = dummy_wrapper(PriceVolume_dr, index_df, index_holdings_path, match_d, constraints, start_dt)
    print("total times is %0.2f"%(time.time()-start))
    generate_basic_stats(aprox, "../../outN", "temp")

