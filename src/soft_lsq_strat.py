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
def after_rebalncing_weights(res1,Ticker2Sector,D_fact):
    D_w_after = {}
    res_d = dict(res1)
    aprx1 = D_tickers[list(res_d.keys())[0]]["return"]*0.0
    total_w = 0
    for r in res_d.keys():
        if Ticker2Sector[r] in D_fact.keys():
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

def wrap_rebalancing(res1,Ticker2Sector,d_weights_sector,bnd):
    D_sector_w = rebalancing_results(dict(res1),Ticker2Sector)
    print(D_sector_w)
    print(d_weights_sector)
    D_fact,D_no_w = updating_rebalancing_weights(d_weights_sector,D_sector_w)
    print("*"*100)
    #print(D_fact)
    aprx1,D_w_after = after_rebalncing_weights(res1,Ticker2Sector,D_fact)
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
    year = 2014
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
    tickers = list(d1["Ticker"])[:100]
    A11 = ["PYPL", "MRNA", "CTVA", "REGN", "VRTX"]
    A11 += ["XTSLA", 'BRKB', 'MMC', 'SYK', "ITW", 'PGR', "PYPL"]
    tickers = [t for t in tickers if t not in A11]
    D_tickers = {x: updt(x, start_dt=start_dt, end_dt=end_dt) for x in tickers}
    #df['your column name'].isnull().sum()
    D_tickers = {x: D_tickers[x] for x in D_tickers.keys() if D_tickers[x]["return"].isnull().sum() ==0}
    d_weights_sector = {x: d_weights_sector[x] for x in d_weights_sector.keys() if x.lower().find("cash") == -1}
    gics_path = os.path.join("..", "data", "GICS", "GICS_sector_SP500.csv")
    Ticker2Sector = prep_ticker2Sector(gics_path)
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
    res_ds = soft_lstsq(D_tickers2, spy["return"].loc[start_dt:end_dt2], lb,ub ,start_dt, end_dt)
    #aprx_ns = create_aprx1(res_ds, D_tickers)
    aprx_ns, D_w_after = wrap_rebalancing(res_ds, Ticker2Sector, d_weights_sector, bnd=4)
    print(np.corrcoef(aprx_ns.loc[start_later:end_dt], spy["return"].loc[start_later:end_dt]))
    d_tmp2 = pd.DataFrame(((1. + aprx_ns).loc[end_dt2:end_dt].cumprod() - 1.))
    d_tmp2["spy_cum_ret"] = ((1. + spy["return"]).loc[end_dt2:end_dt].cumprod() - 1.)
    d_tmp2.plot()
    plt.show()

    print(max(abs(d_tmp2["spy_cum_ret"] - d_tmp2["return"])))
    print(len([x for x in res_ds.keys() if res_ds[x]> 0.0001]))
    print(sum(list(res_ds.values())),sum(D_w_after.values()))
