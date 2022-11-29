import pandas as pd
import os,sys,re
import dateutil
from dateutil.parser import parse
Years_trading_days = 252
Monthly_trading_days = 22
Qtr_trading_days = 60

Constraints_nms = ["Sector_constraints.csv", "forbidden.txt","Value_constraints.csv","forbidden_sectors.txt","Limit.txt"]

def max_dd(L):
    L = list(L)
    mx = L[0]
    n = len(L)
    mx_dd = 0
    for ii in range(n):
        if L[ii] > mx:
            mx = L[ii]
        elif mx_dd < 1 - L[ii]/mx :
            mx_dd = 1 - L[ii]/mx
    return mx_dd

def stat_cols(df):
    close_col = [c for c in df.columns if c.lower().find("close")>-1]
    if len([c for c in close_col if c.lower().find("adj")>-1])>0:
        close_col = [c for c in close_col if c.lower().find("adj")>-1][0]
    else:
        close_col = close_col[0]
    df["return"] = getattr(df[close_col],"pct_change")()
    df["annual_volatility"] = getattr(df["return"].rolling(window = Years_trading_days),"std")()
    df["annual_DrawDown"] = (1. + df["return"]).cumprod().rolling(Years_trading_days).apply(max_dd).tail()
    df["annual_return"]  = df[close_col].pct_change(Years_trading_days).tail()
    df["qtr_volatility"] = getattr(df["return"].rolling(window=Qtr_trading_days), "std")()
    df["qtr_DrawDown"] = (1. + df["return"]).cumprod().rolling(Qtr_trading_days).apply(max_dd).tail()
    df["qtr_return"] = df[close_col].pct_change(Qtr_trading_days).tail()

def input_reader(input_file):
    D_input = {}
    with open(input_file,"r") as input:
        inputs = input.read().split("\n")
        inputs = [x.split(" : ") for x in inputs]

        inputs = {x[0]:x[1] for x in inputs}
    print(inputs, len(list(inputs.items())))
    path_taget_index = os.path.join(*(inputs['Target_index_returns'].split("|")))
    target_index = pd.read_csv(path_taget_index)
    stat_cols(target_index)
    D_input["target_index"] = target_index
    print(target_index.tail())
    PV_data_path = os.path.join(*(inputs["Price_volume_data"].split("|")))
    print(os.listdir(PV_data_path)[:10])
    D_input["PV_data_path"] = PV_data_path
    Target_index_holdings_path = os.path.join(*(inputs["Target_index_holdings"].split("|")))
    D_input["Target_index_holdings_path"] = Target_index_holdings_path
    print(os.listdir(Target_index_holdings_path)[:10])
    print(inputs['Universe'])
    print(inputs['Constraints'])
    print(inputs['RiskFunction'])
    print(inputs["UpdatingDates"])
    updating_dates_path = os.path.join(*(inputs["UpdatingDates"].split("|")))
    with open(updating_dates_path) as dts:
        dts = dts.read().split("\n")
        print([parse(x).strftime("%Y-%m-%d") for x in dts if len(x)==8])
        dts = [parse(x).strftime("%Y-%m-%d") for x in dts if len(x)==8]
        target_index = target_index.set_index("Date")
        print(target_index.loc[dts].tail())
    constraints_path = os.path.join(*(inputs['Constraints'].split("|")))
    constraints = {}
    constraints_files = os.listdir(constraints_path)
    missing = [x for x in Constraints_nms if not x in constraints_files]
    print("missing constraints are: ",missing)
    for f in constraints_files:
        if f[-4:] == ".csv" and f.lower().find("sector") > -1:
            print(f,"\n",pd.read_csv(os.path.join(constraints_path,f)).head())
            constraints["sectors"] = pd.read_csv(os.path.join(constraints_path, f))
            constraints["sectors"] = constraints["sectors"].set_index("Sector")["Max_Weight"].to_dict()

        elif f[-4:] == ".txt" :
            with open(os.path.join(constraints_path,f)) as rd:
                #print(f,"\n",rd.read())
                tickers = (rd.read().split("\n"))
                tickers = [x for x in tickers if re.findall('[A-Z]+',x) and re.findall('[A-Z]+',x)[0] == x]
                print(tickers)
                constraints["forbiden_tickers"] = tickers
    D_input["constraints"] = constraints
    print(D_input)
    return D_input












    
