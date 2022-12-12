from scipy.sparse import rand
from scipy.optimize import lsq_linear
import numpy as np
import time
import os,sys
from dateutil.parser import parse as date_parser
import glob,re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from qpsolvers import solve_qp
from basic_reader import input_reader
from dateutil.parser import parse as date_parse
from dummy_strategy import compute_return
from basic_stats import generate_basic_stats
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

def dates_2_holdings_dict(index_holdings_path):
    print(index_holdings_path)
    L1 = glob.glob(os.path.join(index_holdings_path,"*holdings_*.csv"))
    dts = {re.findall("20[0-9]+", x)[0]:pd.read_csv(x) for x in L1}
    return {date_parser(x).strftime("%Y-%m-%d"):dts[x] for x in dts.keys()}
def create_ticker_to_sector(path_d,dt):
    d2h = dates_2_holdings_dict(path_d)
    #dt = '2019-12-31'
    #print(d2h).keys()
    if not dt in d2h.keys():
        dt = min([x for x in d2h.keys() if x >dt])
    D_strt = d2h[dt].groupby("Sector")["Weight (%)"].sum().to_dict()
    #D_strt = d2h[dt].set_index("Ticker")["Weight (%)"].to_dict()
    return {x:D_strt[x]*0.01 for x in D_strt.keys()}

def prep_ticker2Sector(gics_path):
    #GICS = pd.read_csv(os.path.join("..","Tracker_SP500","data","GICS","GICS_sector_SP500.csv"))
    GICS = pd.read_csv(gics_path)
    Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
    for x in Ticker2Sector.keys():
        if Ticker2Sector[x] == "Health":
            Ticker2Sector[x] = 'Health Care'
        if Ticker2Sector[x] == 'Communication Services':
            Ticker2Sector[x] = 'Communication'
        if Ticker2Sector[x] ==  'Telecommunications Services':
            Ticker2Sector[x] = 'Communication'
    return Ticker2Sector
def prep_sector(Ticker2Sector,tickers):
    Sector2Tickers = {}
    for t in tickers:
        Sector2Tickers[Ticker2Sector[t]] = Sector2Tickers.get(Ticker2Sector[t],[])+[t]
    return Sector2Tickers

def updt(nm,start_dt,end_dt):
    x = pd.read_csv(os.path.join("..","data","PriceVolume",nm+".csv"))
    #print(nm)
    x["return"] = x["Adjusted_close"].pct_change()
    x = x.set_index("Date")
    return x.loc[start_dt:end_dt]


def max_corr(D, tar_ret, start_dt, end_dt=None):
    if end_dt:
        AA = []
        for k in D.keys():
            try:
                AA.append((k, np.corrcoef(D[k]["return"].loc[start_dt:end_dt], tar_ret.loc[start_dt:end_dt])[0, 1]))
            except Exception as e:
                print(e, k)
                print(D[k].loc[start_dt:end_dt].shape)
                print(tar_ret.loc[start_dt:end_dt].shape)
        return max(AA, key=lambda x: x[1])
    else:
        return max([(k, np.corrcoef(D[k]["return"], tar_ret)[0, 1]) for k in D.keys()], key=lambda x: x[1])


def min_diff(D, tar_ret, start_dt, end_dt=None):
    if end_dt:
        AA = []
        for k in D.keys():
            try:
                crr = np.corrcoef(D[k]["return"].loc[start_dt:end_dt], tar_ret.loc[start_dt:end_dt])[0, 1]
                beta = crr * (np.std(tar_ret.loc[start_dt:end_dt]) / np.std(D[k]["return"].loc[start_dt:end_dt]))
                AA.append((k, abs(tar_ret.loc[start_dt:end_dt] - beta * D[k]["return"].loc[start_dt:end_dt]).max()))

            except Exception as e:
                print(e, k)
                print(D[k].loc[start_dt:end_dt].shape)
                print(tar_ret.loc[start_dt:end_dt].shape)
        return min(AA, key=lambda x: x[1])
    else:
        AA = []
        for k in D.keys():
            crr = np.corrcoef(D[k]["return"], tar_ret)[0, 1]
            beta = crr * (np.std(tar_ret) / np.std(D[k]["return"]))
            AA.append((k, np.std(tar_ret - beta * D[k]["return"])))

        return min(AA, key=lambda x: x[1])


def min_var(D, tar_ret, start_dt, end_dt=None):
    if end_dt:
        AA = []
        for k in D.keys():
            try:
                crr = np.corrcoef(D[k]["return"].loc[start_dt:end_dt], tar_ret.loc[start_dt:end_dt])[0, 1]
                beta = crr * (np.std(tar_ret.loc[start_dt:end_dt]) / np.std(D[k]["return"].loc[start_dt:end_dt]))
                AA.append((k, np.std(tar_ret.loc[start_dt:end_dt] - beta * D[k]["return"].loc[start_dt:end_dt])))

            except Exception as e:
                print(e, k)
                print(D[k].loc[start_dt:end_dt].shape)
                print(tar_ret.loc[start_dt:end_dt].shape)
        return min(AA, key=lambda x: x[1])
    else:
        AA = []
        for k in D.keys():
            crr = np.corrcoef(D[k]["return"], tar_ret)[0, 1]
            beta = crr * (np.std(tar_ret) / np.std(D[k]["return"]))
            AA.append((k, np.std(tar_ret - beta * D[k]["return"])))

        return min(AA, key=lambda x: x[1])

def brute_lstsq(D_tickers, tar_ret, start_dt, end_dt):
    D_t_rets = {k: D_tickers[k]["return"].loc[start_dt:end_dt] for k in D_tickers.keys()}
    rets_mat = pd.DataFrame(D_t_rets)
    lsq_sol = np.linalg.lstsq(rets_mat, spy["return"].loc[start_dt:end_dt], rcond=None)[0]
    sol_d = {rets_mat.columns[ii]: lsq_sol[ii] for ii in range(len(rets_mat.columns))}
    return sol_d

def compute_soft_lsq(A,b,lb,ub):
    return lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)['x']
def soft_lstsq(D_tickers, tar_ret, lb, ub, start_dt, end_dt):
    D_t_rets = {k: D_tickers[k]["return"].loc[start_dt:end_dt] for k in D_tickers.keys()}
    rets_mat = pd.DataFrame(D_t_rets)


    #lsq_sol = np.linalg.lstsq(rets_mat, spy["return"].loc[start_dt:end_dt], rcond=None)[0]
    lsq_sol = compute_soft_lsq(rets_mat,tar_ret,lb,ub)
    sol_d = {rets_mat.columns[ii]: lsq_sol[ii] for ii in range(len(rets_mat.columns))}
    return sol_d
def lsq_with_constraints(D_tickersN, tar_ret, lb,ub, start_dt, end_dt,Sector2Tickers,sectors_c={}):
    D_tickers = D_tickersN.copy()
    test_key = list(D_tickers.keys())[0]
    print(list(D_tickers.keys())[0])
    print(D_tickers[list(D_tickers.keys())[0]].shape[0])
    print(tar_ret.shape[0])
    print([k for k in  D_tickers.keys() if "return" not in D_tickers[k].columns])
    relevant_tickers = [k for k in  D_tickers.keys() if "return"  in D_tickers[k].columns]
    #D_t_rets = {k: D_tickers[k]["return"].loc[start_dt:end_dt] for k in relevant_tickers}
    D_t_rets = {k: D_tickers[k]["return"] for k in relevant_tickers}
    print(D_t_rets[test_key].shape)
    tar_ret = tar_ret.loc[start_dt:end_dt]
    print("tar_ret shape", tar_ret.shape)
    rets_mat = pd.DataFrame(D_t_rets)
    rets_mat = rets_mat.loc[tar_ret.index.values]
    print("rets_mat shape ",rets_mat.shape)
    #rets_mat = rets_mat.dropna()
    #print("rets_mat shape after droping rows with nan", rets_mat.shape)
    rets_mat = rets_mat.dropna(axis=1)
    print("rets_mat shape after droping columns with nan", rets_mat.shape)
    rets_mat = rets_mat.dropna()
    print("rets_mat shape after droping rows with nan", rets_mat.shape)


    R = rets_mat.to_numpy()
    s = tar_ret.to_numpy()
    W = np.identity(rets_mat.shape[1])
    print("R.shape",R.shape)
    print("W.shape", W.shape)
    WR = np.dot(R,W)
    P = np.dot(R.transpose(), WR)
    q = -np.dot(s.transpose(), WR)
    G = np.identity(rets_mat.shape[1])
    h = np.array([ub for ii in range(rets_mat.shape[1])])
    Glb = np.identity(rets_mat.shape[1]) * -1
    hlb = np.array([-lb for ii in range(rets_mat.shape[1])])
    Arr = []
    sector_bnd  = []
    if sectors_c:
        for k in sectors_c.keys():
            Sector2Tickers[k]
            A = []
            for t in rets_mat.columns:
                if t in Sector2Tickers[k]:
                    A.append(1)
                else:
                    A.append(0)
            Arr.append(A)
            sector_bnd.append(sectors_c[k])
        G = np.concatenate((G,Glb,np.array(Arr)))
        h = np.array([ub for ii in range(rets_mat.shape[1])]+[-lb for ii in range(rets_mat.shape[1])] + sector_bnd)


    A = np.array([1.0 for ii in range(rets_mat.shape[1])])
    b = np.array([1.0])

    lsq_sol =  solve_qp(P, q, G, h, A, b, solver="cvxopt")
    sol_d = {rets_mat.columns[ii]: lsq_sol[ii] for ii in range(len(rets_mat.columns))}
    return sol_d


def create_aprx1(res_d, D_tickers):
    k = list(D_tickers.keys())[0]
    aprx1 = D_tickers[k]["return"] * 0.0
    total_w = 0
    for r in res_d.items():
        aprx1 += D_tickers[r[0]]["return"] * (r[1])

    return aprx1
def Get_base(D_tickers,tar_ret,metric,mx_p,start_dt,end_dt = None,max_cap = 0.1,fact=0.05):
    D = D_tickers.copy()
    cnt = 0
    bnd = min(mx_p,len(list(D.keys())))
    portfolio = []
    while cnt < bnd:
        if end_dt:
            #print(type(tar_ret))
            t,_ = metric(D,tar_ret,end_dt)
            #print(type(tar_ret))
            #print(type(D[t]["return"]))
            res = D[t].loc[start_dt:end_dt].copy()
            res['one'] = 1.0
            crr = np.corrcoef(tar_ret.loc[start_dt:end_dt],D[t]["return"].loc[start_dt:end_dt])[0,1]
            beta, alpha = np.linalg.lstsq(res[['return', 'one']],
                              tar_ret.loc[start_dt:end_dt],
                              rcond=None)[0]
            #w1 = min(fact*crr*(np.std(tar_ret.loc[start_dt:end_dt])/np.std(D[t]["return"].loc[start_dt:end_dt])),max_cap)
            w1 = min(fact*beta,max_cap)
            #if D_strt.get(t,0.0) < max_cap:
            #w1 = D_strt.get(t,0) #max(w1,D_strt.get(t,0))
            #w1 = min(fact*crr *np.std(D[t]["return"].loc[:end_dt]),max_cap)

            #w1 = min(min(w1,D_strt.get(t,0)),max_cap)
            #w1 = min(crr*np.std(spy["return"]),max_cap)
            print(t,w1)
            #print(t,w1,crr*(np.std(D[t]["return"])))
            tar_ret = tar_ret.loc[start_dt:end_dt] - w1* D[t]["return"].loc[start_dt:end_dt]
        else:
            t,_ = metric(D,tar_ret)
            crr = np.corrcoef(tar_ret,D[t]["return"]).loc[0,1]
            w1 = min(crr*(np.std(tar_ret)/np.std(D[t]["return"])),max_cap)
            w1 = max(w1,0.0)
            #w1 = min(crr*np.std(spy["return"]),max_cap)
            print(t,w1)
            #print(t,w1,crr*(np.std(D[t]["return"])))
            tar_ret = tar_ret - w1* D[t]["return"]
        portfolio.append((t,w1))
        D.pop(t)
        cnt +=1
    return portfolio
def rebalancing_results(res1_d,Ticker2Sector):
    D_sector_w = {}
    for k in res1_d.keys():
        if k in Ticker2Sector.keys():
            D_sector_w[Ticker2Sector[k]] = D_sector_w.get(Ticker2Sector[k],0) + res1_d[k]
    return D_sector_w




# D_sector_w
def updating_rebalancing_weights(d_weights_sector, D_sector_w):
    no_w = []
    D_fact = {}
    for k in d_weights_sector.keys():
        if k in D_sector_w.keys():
            D_fact[k] = d_weights_sector[k] / D_sector_w[k]
        else:
            no_w.append(k)
    print(no_w)
    D_no_w = {}
    for sect in no_w:
        D_no_w[sect] = [k for k in D_tickers.keys() if Ticker2Sector[k] == sect]
    return D_fact, D_no_w
def after_rebalncing_weights(res1,Ticker2Sector,D_fact,D_tickers):
    D_w_after = {}
    res_d = dict(res1)
    aprx1 = D_tickers[list(res_d.keys())[0]]["return"]*0.0
    total_w = 0
    for r in res_d.keys():
        if r in Ticker2Sector.keys() and Ticker2Sector[r] in D_fact.keys():
            aprx1 += D_tickers[r]["return"]*(res_d[r]*D_fact[Ticker2Sector[r]])
            total_w += (res_d[r]*D_fact[Ticker2Sector[r]])
            D_w_after[r] = res_d[r]*D_fact[Ticker2Sector[r]]
    return aprx1,D_w_after
def add_from_missing_sectors(D_no_w,bnd,d_weights_sector,aprx1,D_w_after):
    print("D_no_w",D_no_w)
    for k in D_no_w :
        if D_no_w[k]:
            m1 = min(bnd,len(D_no_w[k]))
            c1 = d_weights_sector[k]/float(m1)
            for ii in range(m1):
                aprx1 += D_tickers[D_no_w[k][ii]]["return"]*(c1)
                #total_w += c1
                D_w_after[D_no_w[k][ii]] = c1
    return aprx1,D_w_after

def wrap_rebalancing(res1,Ticker2Sector,d_weights_sector,D_tickers,bnd):
    D_sector_w = rebalancing_results(dict(res1),Ticker2Sector)
    print(D_sector_w)
    print(d_weights_sector)
    D_fact,D_no_w = updating_rebalancing_weights(d_weights_sector,D_sector_w)
    print("*"*100)
    #print(D_fact)
    aprx1,D_w_after = after_rebalncing_weights(res1,Ticker2Sector,D_fact,D_tickers)
    #print("sm %0.3f"%sum(D_w_after.values()))
    print(rebalancing_results(D_w_after,Ticker2Sector))
    print("-"*55)
    rrr = rebalancing_results(D_w_after,Ticker2Sector)
    print(d_weights_sector)
    for k in d_weights_sector.keys():
        print(k)
        #print(d_weights_sector[k],rrr[k])
    V1 = rebalancing_results(D_w_after,Ticker2Sector)
    print(sum(V1.values()))
    print(sum(d_weights_sector.values()))
    aprx1,D_w_after = add_from_missing_sectors(D_no_w,bnd,d_weights_sector,aprx1,D_w_after)
    print("sm after %0.3f"%sum(D_w_after.values()))
    return aprx1,D_w_after

def compute_lsq(A,b,lb,ub):
    return lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)['x']

"""
rng = np.random.default_rng()

m = 20000
n = 10000

A = np.random.rand(m, n,)
b = rng.standard_normal(m)

lb = rng.standard_normal(n)
ub = lb + 1

res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1)
print(res)
"""
def match_dates(D_tickers_orig,df_tar,target_ret, match_d, d2h,forbidden,sector_bounds,num_of_tickers,ub,lb=0):
    keys_list = list(match_d.keys())
    print(df_tar.index.values[:10])
    print((df_tar.index.values[3]), date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    print((df_tar.index.values[3]) > date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    # df_tar.index = pd.to_datetime(df_tar.index)
    df_tar = df_tar.loc[date_parser(keys_list[0]).strftime("%Y-%m-%d"):]
    print("*" * 50)
    print(keys_list[0], date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    print(match_d.keys())
    print(df_tar.index.values[:10])

    wts_base = d2h[date_parser(keys_list[0]).strftime("%Y-%m-%d")]
    weight_col = [c for c in wts_base.columns if c.lower().find("weight") > -1][0]
    ticker_col = [c for c in wts_base.columns if c.lower().find("ticker") > -1][0]
    for ii in range(len(keys_list) - 1):
        print("match dates ", ii, keys_list[ii])
        D_tickers = D_tickers_orig.copy()
        k = keys_list[ii]
        dt = date_parser(match_d[k]).strftime("%Y-%m-%d")
        # ERROR
        print(d2h[dt])
        year_before = str(int(dt.split("-")[0])-1)
        start_dt = dt#year_before + "-"+dt.split("-")[1]+"-"+dt.split("-")[2]
        end_dt  = date_parser(match_d[keys_list[ii+1]]).strftime("%Y-%m-%d")
        D_tickers2 = {x: D_tickers[x].loc[start_dt:end_dt] for x in D_tickers.keys() if not x in forbidden}
        test_key = list(D_tickers2.keys())[0]
        print(start_dt,end_dt)
        print(test_key)
        print("target_ret")
        print(target_ret.loc[start_dt:end_dt].head())
        print("D_tickers2 ",test_key)
        print(D_tickers2[test_key].head())
        print(target_ret.loc[start_dt:end_dt].shape)
        print(D_tickers2[test_key].shape)

        vals_c = target_ret.loc[start_dt:end_dt].index.values
        D3 = {}
        bad = []
        for x in D_tickers2.keys() :
            try :
                D3[x] = D_tickers2[x].loc[vals_c]
            except Exception as e:
                aaaa = 1+1
        #D_tickers2 = {x:D_tickers2[x].loc[vals_c] for x in D_tickers2.keys() }
        #print("len remain",len(D_tickers2.keys()))
        D3 = {x:D3[x] for x in D3.keys() if D3[x].shape[0] == target_ret.loc[start_dt:end_dt].shape[0]}
        print("len remain", len(D3.keys()))

        res_ds = lsq_with_constraints(D3, target_ret.loc[start_dt:end_dt]["return"], lb, ub, start_dt, end_dt,
                                      Sector2Tickers, sector_bounds)
        # aprx_ns = create_aprx1(res_ds, D_tickers)
        aprx_ns, D_w_after = wrap_rebalancing(res_ds, Ticker2Sector, d_weights_sector, D_tickers,bnd=4)

        d1 = D_w_after
        ks1 = match_d[keys_list[ii + 1]]
        dts1 = date_parser(ks1).strftime("%Y-%m-%d")
        for jj in d1.keys():
            # print(k)
            df_tar[jj].loc[dt:dts1] = d1[jj]

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
        df_tar[k].loc[dt:] = d1[k]
    return df_tar

def wrapper_strategy(PriceVolume_dr,index_df,index_holdings_path,match_d,constraints,start_dt,end_dt,sector_mapping):
    df_tar = create_universe_zero_df(PriceVolume_dr,index_df).loc[start_dt:end_dt]
    d2h = dates_2_holdings_dict(index_holdings_path)
    print(index_df.head(),index_df.tail())
    print(start_dt,end_dt)
    #print(d2h.keys())
    universe = [x.split(".")[0] for x  in os.listdir(PriceVolume_dr) if re.findall('[A-Z]+',x.split(".")[0]) and
                re.findall('[A-Z]+',x.split(".")[0])[0] == x.split(".")[0]]
    close_col = [c for c in index_df.columns if c.lower().find("close") > -1]
    if len([c for c in close_col if c.lower().find("adj") > -1]) > 0:
        close_col = [c for c in close_col if c.lower().find("adj") > -1][0]
    else:
        close_col = close_col[0]
    tickers_pv = create_ret_dict(PriceVolume_dr, universe, close_col)
    #constraints["forbiden_tickers"],constraints["sectors"],constraints["num_of_tickers"],constraints["upper_bound"],sector_mapping)

    forbidden = constraints["forbiden_tickers"]
    sector_bounds = constraints["sectors"]
    num_of_tickers = constraints["num_of_tickers"]
    ub = constraints["upper_bound"]
    match_dates(tickers_pv, df_tar,index_df, match_d, d2h, forbidden, sector_bounds, num_of_tickers, ub, lb=0)

    #match_dates(tickers_pv,df_tar, match_d, d2h, constraints["forbiden_tickers"],constraints["sectors"],constraints["num_of_tickers"],constraints["upper_bound"],sector_mapping)
    universe = list(df_tar.columns)
    close_col = [c for c in index_df.columns if c.lower().find("close") > -1]
    if len([c for c in close_col if c.lower().find("adj") > -1]) > 0:
        close_col = [c for c in close_col if c.lower().find("adj") > -1][0]
    else:
        close_col = close_col[0]

    #filter
    #df_tar = filter_out_forbiden_tickers(df_tar, constraints["forbiden_tickers"])
    aprox = compute_return(df_tar,tickers_pv, start_dt,list(match_d.keys()))
    print("start_dt %s"%(start_dt))
    aprox["benchmark_index_return"] = index_df["return"][start_dt:end_dt]
    aprox["Comulative_ret"] = (1. + aprox["return"][start_dt:end_dt]).cumprod()
    aprox["benchmark_index_comulative_ret"] = (1 + aprox["benchmark_index_return"]).cumprod()
    return aprox,df_tar

GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    """
    start = time.time()
    rng = np.random.default_rng()
    m = 500
    n = 1000
    lb = rng.standard_normal(n)
    ub = lb+1
    A = np.random.rand(m, n, )
    b = rng.standard_normal(m)
    print(compute_lsq(A,b,lb,ub))
    print("total time %0.2f"%(time.time() - start))
    """
    year = 2013
    start_later = "%d-01-01"%year
    start_dt = "%d-01-01"%(year - 3)
    end_dt = "%d-06-30"%year
    path_d = "../data/holdings/IVV/"
    d2h = dates_2_holdings_dict(path_d)
    spy = pd.read_csv("/Users/itaybendan/FinzorAnalytics/SPY.csv")
    spy["return"] = spy["Adjusted_close"].pct_change()
    spy = spy.set_index("Date")
    spy = spy.loc[start_dt:end_dt]
    #print(d2h.keys())
    d_weights_sector = create_ticker_to_sector(path_d, '%d-03-30'%year)
    d1 = pd.read_csv("../data/holdings/IVV/IVV_holdings_20220930_f.csv")
    tickers = list(d1["Ticker"])[:300]
    A11 = ["PYPL", "MRNA", "CTVA", "REGN", "VRTX"]
    A11 += ["XTSLA", 'BRKB', 'MMC', 'SYK', "ITW", 'PGR', "PYPL"]
    tickers = [t for t in tickers if t not in A11]
    D_tickers = {x: updt(x, start_dt=start_dt, end_dt=end_dt) for x in tickers}
    #df['your column name'].isnull().sum()
    D_tickers = {x: D_tickers[x] for x in D_tickers.keys() if D_tickers[x]["return"].isnull().sum() ==0}
    d_weights_sector = {x: d_weights_sector[x] for x in d_weights_sector.keys() if x.lower().find("cash") == -1}
    gics_path = os.path.join("..", "data", "GICS", "GICS_sector_SP500.csv")
    Ticker2Sector = prep_ticker2Sector(gics_path)
    Sector2Tickers = prep_sector(Ticker2Sector,tickers)

    """
    res1 = Get_base(D_tickers, tar_ret=spy["return"].loc[start_dt:end_dt], metric=min_diff, mx_p=50, start_dt=start_dt,
                    end_dt=end_dt, max_cap=0.045)
    aprx1, D_w_after = wrap_rebalancing(res1, Ticker2Sector, d_weights_sector, bnd=4)
    aprx1.tail(), aprx1.head()


    print(np.corrcoef(aprx1.loc[start_dt:end_dt], spy["return"].loc[start_dt:end_dt]))
    """
    #res_d = brute_lstsq(D_tickers, spy["return"].loc[start_dt:end_dt], start_dt, end_dt)
    #aprx_n = create_aprx1(res_d, D_tickers)

    #print(np.corrcoef(aprx_n.loc[start_dt:end_dt], spy["return"].loc[start_dt:end_dt]))
    lb = 0.00
    ub = 0.06
    end_dt2 = "%d-03-30"%year
    D_tickers2 = {x: D_tickers[x].loc[:end_dt2] for x in D_tickers.keys()}
    print(spy["return"].loc[start_dt:end_dt2].shape)
    shp = spy["return"].loc[start_dt:end_dt2].shape[0]
    D_tickers2 = {x: D_tickers2[x] for x in D_tickers2.keys() if D_tickers2[x].shape[0] == shp}
    print(len(D_tickers2.keys()))
    #res_ds = soft_lstsq(D_tickers2, spy["return"].loc[start_dt:end_dt2], lb,ub ,start_dt, end_dt)
    sectors_c = {"Information Technology": 1.,"Financials": 1.05}
    if False:
        res_ds = lsq_with_constraints(D_tickers2, spy["return"].loc[start_dt:end_dt2], lb,ub ,start_dt, end_dt,Sector2Tickers,sectors_c)
        #aprx_ns = create_aprx1(res_ds, D_tickers)
        aprx_ns, D_w_after = wrap_rebalancing(res_ds, Ticker2Sector, d_weights_sector,D_tickers, bnd=4)
        print(aprx_ns.head,"@"*80,aprx_ns.tail())
        #print(np.corrcoef(aprx_ns.loc[start_later:end_dt], spy["return"].loc[start_later:end_dt]))
        d_tmp2 = pd.DataFrame(((1. + aprx_ns).loc[end_dt2:end_dt].cumprod() - 1.))
        d_tmp2["spy_cum_ret"] = ((1. + spy["return"]).loc[end_dt2:end_dt].cumprod() - 1.)
        print(d_tmp2.head())
        d_tmp2.plot()
        plt.show()
        print(D_w_after)

        print(max(abs(d_tmp2["spy_cum_ret"] - d_tmp2["return"])))
        print(len([x for x in res_ds.keys() if res_ds[x]> 0.0001]))
        print(sum(list(res_ds.values())),sum(D_w_after.values()))
        dsffsdfd
    """
    Target_index_returns : ..|data|index_data|SPY.csv
    Target_index_holdings : ..|data|holdings|IVV
    Price_volume_data : ..|data|PriceVolume
    Universe : ..|data|SP500_constiutents_from_2010.csv
    Constraints : ..|Constraints|C2
    UpdatingDates : ..|dates|SP500|rebalancing_dates.txt
    RiskFunction : risk_L2.py
    start_dt : 2020-03-30
    end_dt : 2022-11-01
    Lag : 0
    upper_bound : 0.1
    """
    input_file = os.path.join("..","example","input_files","InputExample.txt")
    D_input = input_reader(input_file)
    print(D_input.keys())
    holdings_files = os.listdir(D_input['index_holdings_path'])
    #logger.info(holdings_files)
    dts = [re.findall("20[0-9]+", x)[0] for x in holdings_files if re.findall("20[0-9]+", x)]
    dts.sort(key=lambda x: date_parse(x))
    lag = D_input["Lag"]
    match_d = {dts[ii]: dts[ii - lag] for ii in range(lag, len(dts))}
    match_d = {k: match_d[k] for k in match_d.keys() if date_parse(k) >= date_parse(D_input["start_dt"])}
    D_input["match_d"] = match_d
    print(match_d.keys())

    D_input["sector_mapping"] = SectorMapping
    D_input["constraints"]["num_of_tickers"] = 200
    D_input["constraints"]["upper_bound"] = ub
    D_input.pop("Lag")
    D_input.pop("upper_bound")
    aprox,df_tar = wrapper_strategy(**D_input)
    out_dir = os.path.join("..","..","lsq_test_res")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    generate_basic_stats(aprox, out_dir, "temp")
    """
    lag = 0
    PriceVolume_dr = os.path.join("..","data","PriceVolume")
    index_df, 
    index_holdings_path = os.path.join("..","data","holdings","IVV")
    match_d, 
    constraints, 
    start_dt, 
    end_dt, 
    PriceVolume_dr, index_df, index_holdings_path, match_d, constraints, start_dt, end_dt, sector_mapping):

    wrapper_strategy()
    """

