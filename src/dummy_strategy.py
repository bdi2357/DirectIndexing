import pandas as pd
import numpy as np
from dates_matching import  dates_2_holdings_dict
#from dates_matching import matching_dates_d2
import os,re,sys
from dateutil.parser import parse as date_parser
#dates_matching

def create_universe_zero_df(PriceVolume_dr,index_df):
    files_paths = os.listdir(PriceVolume_dr)
    #date_col = [c for c in index_df.columns if c.lower()=="date"][0]
    dts = index_df.index.values
    #index_df = index_df.set_index(date_col)
    Universe_Tickers = [x.split(".")[0] for x in files_paths
                        if re.findall('[A-Z]+',x.split(".")[0]) and
                        x.split(".")[0] == re.findall('[A-Z]+',x.split(".")[0])[0]]
    big_df = pd.DataFrame(columns=Universe_Tickers, index=dts)
    return big_df.fillna(0.0)


def match_dates(df_tar, match_d, d2h,forbidden,sector_bounds,num_of_tickers,upper_bound,sector_mapping):
    keys_list = list(match_d.keys())
    print(df_tar.index.values[:10])
    print((df_tar.index.values[3]),date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    print((df_tar.index.values[3]) > date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    #df_tar.index = pd.to_datetime(df_tar.index)
    df_tar = df_tar.loc[date_parser(keys_list[0]).strftime("%Y-%m-%d"):]
    print("*"*50)
    print(keys_list[0],date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    print(match_d.keys())
    print(df_tar.index.values[:10])
    
    wts_base = d2h[date_parser(keys_list[0]).strftime("%Y-%m-%d")]
    weight_col = [c for c in wts_base.columns if c.lower().find("weight") > -1][0]
    ticker_col = [c for c in wts_base.columns if c.lower().find("ticker") > -1][0]
    for ii in range(len(keys_list) - 1):
        print("match dates ",ii,keys_list[ii])
        k = keys_list[ii]
        dt = date_parser(match_d[k]).strftime("%Y-%m-%d")
        weights = d2h[dt].set_index('Ticker')['Weight (%)']
        eligible_tickers = list(df_tar.columns)
        eligible = [c for c in weights.index.values if c in eligible_tickers]
        weights = weights.loc[eligible]
        sm1 = weights.sum()
        weights *= 1. / sm1
        d1 = weights.to_dict()
        d1 = filter_weights_dict(d1,forbidden)
        print(k,dt)
        d1 = limit_tickers(d1,num_of_tickers)
        #print("AAPL before", weights["AAPL"])
        d1 = filter_weights_dict_sector_weights(d1, sector_bounds, sector_mapping)
        #print("AAPL after", weights["AAPL"])
        d1 = max_cap_ticker(d1,upper_bound)
        ks1 = match_d[keys_list[ii + 1]]
        dts1 = date_parser(ks1).strftime("%Y-%m-%d")
        for jj in d1.keys():
            #print(k)
            df_tar[jj].loc[dt:dts1] = d1[jj]

    k = match_d[keys_list[-1]]
    
    f_cands = os.listdir(os.path.join("..","data","holdings","IVV"))
    f_cands = [x for x in f_cands if x.find("IVV_holdings")>-1]
    sol1 = max([x for x in f_cands if x< 'IVV_holdings_%s_f.csv'%k])
    kd = re.findall('20[0-9]+',sol1)[0]
    dt = date_parser(kd).strftime("%Y-%m-%d")
    weights = d2h[dt].set_index(ticker_col)[weight_col]
    eligible_tickers = list(df_tar.columns)
    eligible = [c for c in weights.index.values if c in eligible_tickers]
    weights = weights.loc[eligible]
    sm1 = weights.sum()
    weights *= 1. / sm1
    d1 = weights.to_dict()
    for k in d1.keys():
        df_tar[k].loc[dt:] = d1[k]
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
            print("in create_ret_dict")
            bad_tickers.append(k)
    return tickers_pv


def compute_return(df_tar,tickers_pv,start_dt,dates):
    print(dates)
    print(start_dt)
    df_tar = df_tar[start_dt:]
    ret_aprox = {}
    """
    FIX ###
    """
    df_tar = df_tar[date_parser(dates[0]).strftime("%Y-%m-%d"):]
    vals = list(df_tar.index.values)
    cr = 0
    dts_vals = {}
    tmp = vals[0]
    print("Here1")
    for ii in range(len(dates)-1):
        #tmp = vals[cr]
        dts_vals[dates[ii]] = []
        while tmp < dates[ii+1] and cr < len(vals):
            tmp = vals[cr]
            cr+=1
            dts_vals[dates[ii]].append(tmp)

    dts_vals[dates[-1]] = []
    print("Here2")
    while cr < len(vals):
        tmp = vals[cr]
        dts_vals[dates[-1]].append(tmp)
        cr+=1

    print("Here3")
    for ii in range(len(dates)):
        for dt in dts_vals[dates[ii]]:
            print(dt)
            sm = 0
            miss = 0
            wrong = []
            if dt == "2020-04-01":
                print(k, df_tar.loc[dt, k], tickers_pv[k].loc[dt, "return"])
            if dts_vals[dates[ii]].index(dt) == 0:
                for k in tickers_pv.keys():


                    if df_tar.loc[dt, k] > 0:
                        try:
                            #df_tar.loc[dt, k]
                            if np.isnan(tickers_pv[k].loc[dt, "return"]) or np.isnan(df_tar.loc[dt, k]):
                                print(k,dt,"ISNAN")
                                wrong.append(k)
                                miss += df_tar.loc[dt, k]
                            else:
                                sm += tickers_pv[k].loc[dt, "return"] * df_tar.loc[dt, k]
                        except Exception as e:
                            print(e,k)
                            print(k in df_tar.columns)
                            print("*"*40)
                            wrong.append(k)
                            miss += df_tar.loc[dt, k]
                ret_aprox[dt] = sm / (1. - miss)
                prev = dt
            else:
                for k in tickers_pv.keys():

                    if df_tar.loc[dt, k] > 0:
                        try:
                            #df_tar.loc[dt, k] =  df_tar.loc[prev, k]*(1. + tickers_pv[k].loc[prev, "return"] )
                            if np.isnan(tickers_pv[k].loc[dt, "return"]) or np.isnan(df_tar.loc[dt, k]):
                                print(k, dt, "ISNAN")
                                wrong.append(k)
                                miss += df_tar.loc[dt, k]
                            else:
                                sm += tickers_pv[k].loc[dt, "return"] * df_tar.loc[dt, k]
                        except Exception as e:
                            print(e, k)
                            print(k in df_tar.columns)
                            print(dt in tickers_pv[k].index.values)
                            print(dt in df_tar.index.values)
                            print("=" * 40)
                            wrong.append(k)
                            miss += df_tar.loc[dt, k]
                ret_aprox[dt] = sm / (1. - miss)
                prev = dt


    aprox1 = pd.DataFrame(ret_aprox.items(), columns=["Date", "return"])
    aprox1 = aprox1.set_index("Date")
    return aprox1
def filter_out_forbiden_tickers(df_tar,tickers):
    sm = df_tar.apply(lambda r: 1./(1-sum([r[t] for t in tickers])),axis=1)
    for t in tickers:
        df_tar[t] = 0
    func = lambda x: np.asarray(x) * np.asarray(sm)

    return df_tar.apply(func)

def filter_weights_dict(weights,forbidden):
    #print(weights.keys())
    #print("WFC" in weights.keys())
    forbidden_weight = sum([weights[k] for k in forbidden if k in weights.keys()])
    total_sum = sum([weights[k] for k in weights.keys()])
    for k in forbidden:
        if k in weights.keys():
            weights.pop(k)
    fctr = (total_sum/(total_sum - forbidden_weight))
    return {k:weights[k] * fctr  for k in weights.keys()}


def filter_weights_dict_sector_weights(weights, sector_bounds, sector_mapping):
    print("IN filter_weights_dict_sector_weights")
    diffs = 0
    for k in sector_bounds.keys():
        sm = sum([weights[ticker] for ticker in sector_mapping[k] if ticker in weights.keys()])
        if sm > sector_bounds[k]:
            dif = sm - sector_bounds[k]
            for ticker in sector_mapping[k]:
                if ticker in weights.keys():
                    weights[ticker] *= sector_bounds[k] / sm
            diffs += dif
    sum_others = 0
    for k in sector_mapping.keys():
        if not k in sector_bounds:
            sum_others += sum([weights[ticker] for ticker in sector_mapping[k] if ticker in weights.keys()])
    fact = (sum_others + diffs) / sum_others
    for k in sector_mapping.keys():
        if not k in sector_bounds:
            for ticker in sector_mapping[k]:
                if ticker in weights.keys():
                    weights[ticker] *= fact

    return weights  # {k:weights[k] for k in weights.keys()}

def limit_tickers(weights,maximal_num):
    maximal_num = min(maximal_num,len(list(weights.keys())))
    w_list = list(weights.items())
    w_list.sort(key = lambda x:x[1],reverse = True)
    d_out  = dict(w_list[:maximal_num])
    sm = sum(d_out.values())
    d_out = {k:d_out[k]/sm for k in d_out.keys()}
    return d_out

def max_cap_ticker(weights,upper_bound):
    abv = []
    blw = []
    sm = 0
    sm_blw = 0
    for k in weights.keys():
        if weights[k] > upper_bound:
            sm += weights[k]- upper_bound
            weights[k] -= weights[k]- upper_bound
            abv.append(k)
        else:
            blw.append(k)
            sm_blw +=weights[k]
    fact = 1. + (sm/sm_blw)
    for k in blw:
        weights[k] *= fact
    return weights


def wrapper_strategy(PriceVolume_dr,index_df,index_holdings_path,match_d,constraints,start_dt,end_dt,sector_mapping):
    df_tar = create_universe_zero_df(PriceVolume_dr,index_df)
    d2h = dates_2_holdings_dict(index_holdings_path)
    #print(d2h.keys())
    match_dates(df_tar, match_d, d2h, constraints["forbiden_tickers"],constraints["sectors"],constraints["num_of_tickers"],constraints["upper_bound"],sector_mapping)
    universe = list(df_tar.columns)
    close_col = [c for c in index_df.columns if c.lower().find("close") > -1]
    if len([c for c in close_col if c.lower().find("adj") > -1]) > 0:
        close_col = [c for c in close_col if c.lower().find("adj") > -1][0]
    else:
        close_col = close_col[0]
    tickers_pv = create_ret_dict(PriceVolume_dr, universe, close_col)
    #filter
    #df_tar = filter_out_forbiden_tickers(df_tar, constraints["forbiden_tickers"])
    aprox = compute_return(df_tar,tickers_pv, start_dt,list(match_d.keys()))
    print("start_dt %s"%(start_dt))
    aprox["benchmark_index_return"] = index_df["return"][start_dt:end_dt]
    aprox["Comulative_ret"] = (1. + aprox["return"][start_dt:end_dt]).cumprod()
    aprox["benchmark_index_comulative_ret"] = (1 + aprox["benchmark_index_return"]).cumprod()
    return aprox,df_tar







"""
def dummy_strat_index(index_returns,rebalncing_dates,index_holdings_path,constraints):
    rebalncing_dates_dict = {rebalncing_dates[ii] : rebalncing_dates[ii-1] for ii in range(1,len(rebalncing_dates))}
    d2h = dates_2_holdings_dict(index_holdings_path)
    matching_d = matching_dates_d(list(rebalncing_dates_dict.keys()),list(d2h.keys()))
    weights_d = {d: d2h[matching_d[d]] for d in rebalncing_dates_dict.keys()}
"""

if __name__ == "__main__":
    print("In")


