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

def create_ret_dict(PriceVolume_dr,universe,close_col):
    #close_col = "Adjusted_close"
    tickers_pv = {t: pd.read_csv(os.path.join(PriceVolume_dr, t + ".csv")) for t in universe}
    bad_tickers = []
    for k in tickers_pv.keys():
        try:
            tickers_pv[k][close_col] = tickers_pv[k][close_col].fillna(0.00)
            tickers_pv[k]["return"] = getattr(tickers_pv[k][close_col], "pct_change")()
            tickers_pv[k] = tickers_pv[k].set_index("Date")
        except Exception as e:
            print(e, " ", k)
            bad_tickers.append(k)
    return tickers_pv


def compute_return(df_tar,tickers_pv,start_dt):
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
def filter_out_forbiden_tickers(df_tar,tickers):
    sm = df_tar.apply(lambda r: 1./(1-sum([r[t] for t in tickers])),axis=1)
    for t in tickers:
        df_tar[t] = 0
    func = lambda x: np.asarray(x) * np.asarray(sm)

    return df_tar.apply(func)

def dummy_wrapper(PriceVolume_dr,index_df,match_d,constraints,start_dt):
    tar_df = create_universe_zero_df(PriceVolume_dr,index_df)
    d2h = dates_2_holdings_dict(index_holdings_path)
    match_dates(df_tar, match_d, d2h)
    tickers_pv = create_ret_dict(PriceVolume_dr, universe, close_col)
    df_tar = filter_out_forbiden_tickers(df_tar, constraints["forbiden_tickers"])
    aprox = compute_return(df_tar,tickers_pv, start_dt)
    aprox["benchmark_index_return"] = index_df["return"][start_dt:]
    aprox["Comulative_ret"] = (1. + aprox["return"]).cumprod()
    aprox["benchmark_index_comulative_ret"] = (1 + index_df["return"][start_dt:]).cumprod()
    return aprox






"""
def dummy_strat_index(index_returns,rebalncing_dates,index_holdings_path,constraints):
    rebalncing_dates_dict = {rebalncing_dates[ii] : rebalncing_dates[ii-1] for ii in range(1,len(rebalncing_dates))}
    d2h = dates_2_holdings_dict(index_holdings_path)
    matching_d = matching_dates_d(list(rebalncing_dates_dict.keys()),list(d2h.keys()))
    weights_d = {d: d2h[matching_d[d]] for d in rebalncing_dates_dict.keys()}
"""



