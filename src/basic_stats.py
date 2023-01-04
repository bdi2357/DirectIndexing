import numpy as np
import os,sys,re
import pandas as pd
import matplotlib.pyplot as plt
from basic_reader import max_dd
from risk_L2 import Risk

def get_sector_weights(df_all):
    GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
    Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
    SectorMapping = {}
    for k in Ticker2Sector.keys():
        SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
    add_h = SectorMapping.pop("Health")
    SectorMapping["Health Care"] += add_h
    Dx = []
    for k in SectorMapping.keys():
        Dx.append(pd.DataFrame(df_all[[x for x in df_all.columns if x in SectorMapping[k]]].sum(axis=1),columns = [k]))
    sector_weights = pd.concat(Dx,axis=1)
    sector_weights = sector_weights.drop_duplicates(subset= ['Health Care', 'Financials', 'Information Technology'],keep="first")
    return sector_weights

def generate_basic_stats(df,output_dir,name):
    stats = {}
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    df = df[1:-1]
    df["benchmark_index_comulative_ret"] = (1.+ df["benchmark_index_return"]).cumprod() - 1.
    df["Comulative_ret"] = (1. + df["return"]).cumprod() - 1.
    df[[c for c in df.columns if c.lower().find("comulative")>-1]].plot()
    plt.savefig(os.path.join(output_dir,'benchmarkVstracker.png'))
    stats["maximum_abs_difference"] = max(abs(df["benchmark_index_comulative_ret"] - df["Comulative_ret"]))
    stats["anuallized_volatility"] = (df["benchmark_index_return"]-df["return"]).std() *(df.shape[0]**0.5)
    stats["max_dd"] = max(max_dd(1. + df["benchmark_index_comulative_ret"] - df["Comulative_ret"] ),max_dd(1. -df["benchmark_index_comulative_ret"] + df["Comulative_ret"] ))
    out_df = pd.DataFrame(stats.items(),columns= ["Stat","Value"])
    out_df.to_csv(os.path.join(output_dir, 'stats.csv'))
    #date_col = [c for c in df.columns if c.lower().find("date") > -1][0]
    df["year"] = df.apply(lambda r: r.name[:4], axis=1)
    years = sorted(list(set(df["year"])))
    D_years = {k: v for k, v in df.groupby("year")}
    by_year = {}
    for y in years:
        ret_y = ((D_years[y]["return"] + 1.).cumprod() - 1.)
        benchmark_ret_y = ((D_years[y]["benchmark_index_return"]+1.).cumprod() -1. )
        by_year[y] = abs(ret_y-benchmark_ret_y).max()
    pd.DataFrame(by_year.items(), columns=["Year", "MaxAbsDiff"]).to_csv(os.path.join(output_dir, 'by_year.csv'))
    if not os.path.isdir(os.path.join(output_dir,"raw")):
        os.mkdir(os.path.join(output_dir,"raw"))
    df.to_csv(os.path.join(output_dir,"raw","basic_data.csv"))


if __name__ == "__main__":
    print("In")