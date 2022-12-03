import logging
import os,sys,re
import numpy as np
import pandas as pd
from dummy_strategy import dummy_wrapper
import time
import glob
from basic_stats import generate_basic_stats
from dateutil.parser import parse as date_parse
import argparse

GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    path_d = os.path.join("..","data","holdings","IVV")
    PriceVolume_dr = os.path.join("..","data","PriceVolume")
    #index_df = pd.read_csv(os.path.join("..","data","index_data","SPY.csv"))
    index_df = pd.read_csv(os.path.join("..", "data", "index_data", "IVV.csv"))
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
    #match_d = {dts[ii]: dts[ii -1] for ii in range(1, len(dts))}
    lag = 0
    match_d = {dts[ii]: dts[ii -lag ] for ii in range(lag, len(dts))}
    constraints = {}
    """
    constraints["forbiden_tickers"] = ["MSFT","XOM","BAC","JPM","WFC","AXP","AAPL","NVDA",
                                       "V","AXP","MA","WMT","T","JNJ","BSX","BAC","AMZN",
                                       "GOOG","INTC","AMD"] +forbiden_const
    """
    forbiden_const = ["T"]
    constraints["forbiden_tickers"] =  forbiden_const
    constraints["sectors"] = {'Information Technology':0.6,'Consumer Discretionary':115}
    constraints["num_of_tickers"] = 600
    constraints["upper_bound"] = 0.03
    sector_mapping = SectorMapping
    start_dt = '2021-03-30'
    end_dt = '2022-11-01'
    match_d = {k: match_d[k] for k in match_d.keys() if k >= start_dt}
    #print(match_d.keys())
    print(match_d.keys())
    #exit(0)
    start = time.time()
    index_holdings_path = os.path.join("..","data","holdings","IVV")
    aprox,df_tar = dummy_wrapper(PriceVolume_dr, index_df, index_holdings_path, match_d, constraints, start_dt,end_dt,sector_mapping)
    print("total times is %0.2f"%(time.time()-start))

    output_dirs = os.path.join("..","..","hyper_params")
    if not os.path.isdir(output_dirs):
        os.mkdir(output_dirs)
    strat_name = "dummy"
    strat_output_dir = os.path.join(output_dirs,strat_name)
    if not os.path.isdir(strat_output_dir):
        os.mkdir(strat_output_dir)
    out_dir = os.path.join(strat_output_dir, "tmp1")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    generate_basic_stats(aprox, out_dir, "temp")
    aprox.to_csv(os.path.join(out_dir,"aprox.csv"))