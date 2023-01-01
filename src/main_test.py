import logging
from basic_reader import input_reader
import os,sys,re
import numpy as np
import pandas as pd
#from dummy_strategy import dummy_wrapper
import time
import glob
from basic_stats import generate_basic_stats
from dateutil.parser import parse as date_parse
import argparse
import importlib


logging.basicConfig(filename = 'file.log',
                    level = logging.WARNING,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("color")
logger.setLevel(logging.DEBUG)
shandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
shandler.setFormatter(formatter)
logger.addHandler(shandler)

GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    start = time.time()
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--InputFile', action='store', type=str, required=True)
    my_parser.add_argument('--TrackerModule', action='store', type=str, required=True)
    my_parser.add_argument('--OutDir', action='store', type=str, required=True)

    args = my_parser.parse_args()

    print(args.InputFile)
    print(args.TrackerModule)
    print(args.OutDir)
    strat_name = args.TrackerModule.split(".")[0]
    print(strat_name)
    module = importlib.import_module(strat_name, package=None)
    #exit(0)
    #input_file = os.path.join("..","example","input_files","InputExample.txt")
    input_file = args.InputFile
    D_input = input_reader(input_file)
    holdings_files = os.listdir(D_input['index_holdings_path'])
    logger.info(holdings_files)
    dts = [re.findall("20[0-9]+", x)[0] for x in holdings_files if re.findall("20[0-9]+", x)]
    dts.sort(key = lambda x: date_parse(x))
    lag = D_input["Lag"]
    upper_bound = D_input["upper_bound"]
    if args.TrackerModule.lower().find("dummy") > -1:
        match_d = {dts[ii]: dts[ii - lag] for ii in range(lag, len(dts))}
    else:
        print("HERE")
        match_d = {dts[ii]: dts[ii] for ii in range(lag, len(dts))}
    
    L_bef = list(match_d.items())
    L_bef.sort(key = lambda x: x[0])
    print(L_bef)
    print("+"*40)
    match_d = dict(L_bef)
    match_d = {k: match_d[k] for k in match_d.keys() if date_parse(k) >= date_parse(D_input["start_dt"])}
    match_d[D_input["start_dt"].replace("-","")] = D_input["start_dt"].replace("-","")
    
    L = list(match_d.items())
    L.sort(key = lambda x: x[1])
    print(L)
    
    D_input["match_d"] = match_d
    print(list(match_d.keys()))
    print("======")
    print(match_d.items())
   
    D_input["sector_mapping"] =  SectorMapping
    D_input["constraints"]["num_of_tickers"] = D_input["num_of_tickers"]
    D_input["constraints"]["upper_bound"] = upper_bound
    D_input.pop("Lag")
    D_input.pop("upper_bound")
    D_input.pop("num_of_tickers")
    logger.info(type(list(match_d.keys())[0]))
    #print(D_input["index_df"].index.values[:10])
    aprox,df_tar = module.wrapper_strategy(**D_input)
    
    logger.info("total times is %0.2f" % (time.time() - start))
    out_dir = args.OutDir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(os.path.join(out_dir,"raw")):
        os.mkdir(os.path.join(out_dir,"raw"))
    df_tar[D_input["start_dt"]:D_input["end_dt"]].to_csv(os.path.join(out_dir,"raw","dummy_weights_file.csv"))
    generate_basic_stats(aprox, out_dir, "temp")
    aprox.to_csv(os.path.join(out_dir, "aprox.csv"))