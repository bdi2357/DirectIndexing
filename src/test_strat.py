import os,sys,re
import numpy as np
import pandas as pd
from dummy_strategy import dummy_wrapper
import time
import glob
from basic_stats import generate_basic_stats

forbiden_const = ['BXLT',
 'SHLDQ',
 'PENN',
 'FHI',
 'MTZ',
 'VNT',
 'CPPRQ',
 'ALD',
 'DEN',
 'MMI',
 'NCRA',
 'AABA',
 'INCO',
 'TMUSR',
 'MNKKQ',
 'DOFSQ',
 'CBE',
 'WPX',
 'XTSLA',
 'UBFUT',
 'CMCSK',
 'USD',
 'KTF',
 'PETM',
 'GHC',
 'MPN',
 'CPGX',
 'BEAM',
 'RSHCQ',
 'GEC',
 'ANRZQ',
 'CPRI',
 'OSHWQ',
 'NEBLQ']
sector_df = pd.read_csv(os.path.join("data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    path_d = os.path.join("..","data","holdings","IVV")
    PriceVolume_dr = os.path.join("..","data","PriceVolume")
    index_df = pd.read_csv(os.path.join("..","data","index_data","SPY.csv"))
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
    constraints["forbiden_tickers"] = ["MSFT","XOM","BAC","JPM","WFC","AXP","AAPL","NVDA",
                                       "V","AXP","MA","WMT","T","JNJ","BSX","BAC","AMZN",
                                       "GOOG","INTC","AMD"] +forbiden_const
    start_dt = '2021-09-30'
    start = time.time()
    index_holdings_path = os.path.join("..","data","holdings","IVV")
    aprox = dummy_wrapper(PriceVolume_dr, index_df, index_holdings_path, match_d, constraints, start_dt)
    print("total times is %0.2f"%(time.time()-start))
    generate_basic_stats(aprox, "../../outN5", "temp")

