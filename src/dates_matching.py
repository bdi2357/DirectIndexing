import pandas as pd
import numpy as np
import os,sys,re
import glob
from dateutil.parser import parse as date_parser
def dates_2_holdings_dict(index_holdings_path):
    L1 = glob.glob(os.path.join(index_holdings_path,"*holdings_*.csv"))
    dts = {re.findall("20[0-9]+", x)[0]:pd.read_csv(x) for x in L1}
    return {date_parser(x).strftime("%Y-%m-%d"):dts[x] for x in dts.keys()}
def matching_d2(rebalncing_dates,holdings_dates):
    rd = [(x,'rd') for x in rebalncing_dates]
    hd = [(x,'hd') for x in holdings_dates]
    union = rd+hd
    union.sort()
    match_d = {}
    for ii  in range(len(union)):
        if union[ii][1] == 'rd':
            jj = ii-1
            while jj >=0:
                if union[jj][1] == 'hd':
                    match_d[union[ii][0]] = union[jj][0]
                    break;
                jj-=1
    return match_d

if __name__ =="__main__":
    print("In")

