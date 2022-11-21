import pandas as pd
from dates_matching import  dates_2_holdings_dict,matching_dates_d
import os,re,sys
#dates_matching

def create_universe_zero_df(PriceVolume_dr,index_df):
    files_paths = os.listdir(PriceVolume_dr)
    date_col = [c for c in index_df.columns if c.lower()=="date"][0]
    dts = index_df[date_col]
    Universe_Tickers = [x.split(".")[0] for x in files_paths
                        if re.findall('[A-Z]+',x.split(".")[0]) and
                        x.split(".")[0] == re.findall('[A-Z]+',x.split(".")[0])[0]]
    big_df = pd.DataFrame(columns=Universe_Tickers, index=dts)
    return big_df.fillna(0.0)


def match_dates(df_tar, match_d, d2h):
    keys_list = list(match_d.keys())
    df_tar = df_tar[keys_list[0]:]
    wts_base = d2h[date_parser(keys_list[0]).strftime("%Y-%m-%d")]
    weight_col = [c for c in wts_base.columns if c.lower().find("weight") > -1][0]
    ticker_col = [c for c in wts_base.columns if c.lower().find("ticker") > -1][0]
    for ii in range(len(keys_list) - 1):
        k = keys_list[ii]
        dt = date_parser(match_d[k]).strftime("%Y-%m-%d")
        weights = d2h[dt].set_index('Ticker')['Weight (%)']
        eligible_tickers = list(big_df_trm.columns)
        eligible = [c for c in weights.index.values if c in eligible_tickers]
        weights = weights.loc[eligible]
        sm1 = weights.sum()
        weights *= 1. / sm1
        d1 = weights.to_dict()

        ks1 = match_d[keys_list[ii + 1]]
        dts1 = date_parser(ks1).strftime("%Y-%m-%d")
        for k in d1.keys():
            df_tar[dt:dts1][k] = d1[k]

    k = match_d[keys_list[-1]]
    dt = date_parser(k).strftime("%Y-%m-%d")
    weights = d2h[dt].set_index(ticker_col)[weight_col]
    eligible_tickers = list(df_tar.columns)
    eligible = [c for c in weights.index.values if c in eligible_tickers]
    weights = weights.loc[eligible]
    sm1 = weights.sum()
    weights *= 1. / sm1
    d1 = weights.to_dict()
    for k in d1.keys():
        df_tar.loc[dt:][k] = d1[k]
    return df_tar


def compute_return(df_tar, start_dt):
    df_tar = df_tar[start_dt:]
    ret_aprox = {}
    for dt in df_tar.index.values[:]:
        sm = 0
        miss = 0
        wrong = []
        for k in tickers_pv.keys():

            if df_tar.loc[dt, k] > 0:
                try:
                    sm += tickers_pv[k].loc[dt, "return"] * df_tar.loc[dt, k]
                except:
                    wrong.append(k)
                    miss += df_tar.loc[dt, k]
        ret_aprox[dt] = sm / (1. - miss)
    aprox1 = pd.DataFrame(ret_aprox.items(), columns=["Date", "return"])
    aprox1 = aprox1.set_index("Date")
    return aprox1


def dummy_strat_index(index_returns,rebalncing_dates,index_holdings_path,constraints):
    rebalncing_dates_dict = {rebalncing_dates[ii] : rebalncing_dates[ii-1] for ii in range(1,len(rebalncing_dates))}
    d2h = dates_2_holdings_dict(index_holdings_path)
    matching_d = matching_dates_d(list(rebalncing_dates_dict.keys()),list(d2h.keys()))
    weights_d = {d: d2h[matching_d[d]] for d in rebalncing_dates_dict.keys()}




