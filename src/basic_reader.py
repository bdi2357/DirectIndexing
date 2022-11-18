import pandas as pd
import os,sys,re
import dateutil
from dateutil.parser import parse
Years_trading_days = 252
Monthly_trading_days = 22
Qtr_trading_days = 60

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
    with open(input_file,"r") as input:
        inputs = input.read().split("\n")
        inputs = [x.split(" : ") for x in inputs]

        inputs = {x[0]:x[1] for x in inputs}
    print(inputs, len(list(inputs.items())))
    target_index = pd.read_csv(inputs['Target_index_returns'])
    stat_cols(target_index)
    print(target_index.tail())
    print(os.listdir(inputs["Price_volume_data"])[:10])
    print(os.listdir(inputs["Target_index_holdings"])[:10])
    print(inputs['Universe'])
    print(inputs['Constraints'])
    print(inputs['RiskFunction'])
    print(inputs["UpdatingDates"])
    with open(inputs["UpdatingDates"]) as dts:
        dts = dts.read().split("\n")
        print([parse(x).strftime("%Y-%m-%d") for x in dts if len(x)==8])
        dts = [parse(x).strftime("%Y-%m-%d") for x in dts if len(x)==8]
        target_index = target_index.set_index("Date")
        print(target_index.loc[dts].tail())








    
