from basic_reader import input_reader
import os,sys,re
import numpy as np
import pandas as pd
from dummy_strategy import dummy_wrapper
import time
import glob
from basic_stats import generate_basic_stats


GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    input_file = os.path.join("..","example","input_files","InputExample.txt")
    D_input = input_reader(input_file)
    holdings_files = os.listdir(D_input['index_holdings_path'])
    print(holdings_files)
    dts = [re.findall("20[0-9]+", x)[0] for x in holdings_files if re.findall("20[0-9]+", x)]
    lag = D_input["Lag"]
    match_d = {dts[ii]: dts[ii - lag] for ii in range(lag, len(dts))}
    D_input["match_d"] = match_d
    D_input["sector_mapping"] =  SectorMapping
    D_input["constraints"]["num_of_tickers"] = 600
    D_input.pop("Lag")
    aprox = dummy_wrapper(**D_input)
    print("total times is %0.2f" % (time.time() - start))
    out_dir = "../../outN34"
    generate_basic_stats(aprox, out_dir, "temp")
    aprox.to_csv(os.path.join(out_dir, "aprox.csv"))