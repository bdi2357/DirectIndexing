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
        if t in Ticker2Sector.keys():
            Sector2Tickers[Ticker2Sector[t]] = Sector2Tickers.get(Ticker2Sector[t],[])+[t]
    return Sector2Tickers
PriceVolume_dr = os.path.join("..","data","PriceVolume")
files_paths = os.listdir(PriceVolume_dr)
tickers = [x.split(".")[0] for x in files_paths
                        if re.findall('[A-Z]+',x.split(".")[0]) and
                        x.split(".")[0] == re.findall('[A-Z]+',x.split(".")[0])[0]]

gics_path = os.path.join("..", "data", "GICS", "GICS_sector_SP500.csv")
Ticker2Sector = prep_ticker2Sector(gics_path)
Sector2Tickers = prep_sector(Ticker2Sector,tickers)
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
def cum_prd(srs):
    return (srs+1.).cumprod() #- 1. #srs #(srs+1.).cumprod()#srs
def cum_prd2(srs):
    return srs #.apply(lambda r: r*r*r) #(srs+1.).cumprod() #- 1.
def lsq_with_constraints(D_tickersN, tar_ret, lb,ub, start_dt, end_dt,Sector2Tickers,sectors_c={}):
    D_tickers = D_tickersN.copy()
    test_key = list(D_tickers.keys())[0]
    print(list(D_tickers.keys())[0])
    print(D_tickers[list(D_tickers.keys())[0]].shape[0])
    print(tar_ret.shape[0])
    print([k for k in  D_tickers.keys() if "return" not in D_tickers[k].columns])
    relevant_tickers = [k for k in  D_tickers.keys() if "return"  in D_tickers[k].columns]
    #D_t_rets = {k: D_tickers[k]["return"].loc[start_dt:end_dt] for k in relevant_tickers}
    #D_t_rets2 = {k: cum_prd(D_tickers[k]["return"]) for k in relevant_tickers}
    D_t_rets = {k: cum_prd2(D_tickers[k]["return"]) for k in relevant_tickers}
    print(D_t_rets[test_key].shape)
    #tar_ret2 = cum_prd(tar_ret.loc[start_dt:end_dt].copy())
    tar_ret = cum_prd2(tar_ret.loc[start_dt:end_dt])
    print("tar_ret shape", tar_ret.shape)
    rets_mat = pd.DataFrame(D_t_rets)
    print(rets_mat.index.values[:10])
    print(tar_ret.index.values[:10])
    rets_mat = rets_mat.loc[tar_ret.index.values]
    #rets_mat2 = pd.DataFrame(D_t_rets)
    #rets_mat2 = rets_mat2.loc[tar_ret.index.values]
    #tar_ret = pd.concat([tar_ret,tar_ret2])
    #rets_mat = pd.concat([rets_mat,rets_mat2])
    print("rets_mat shape ",rets_mat.shape)
    print(rets_mat.index.values)
    
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
    if isinstance(ub,float):
        h = np.array([ub for ii in range(rets_mat.shape[1])])
    else:
        h = np.array(ub)
    Glb = np.identity(rets_mat.shape[1]) * -1
    if isinstance(lb,float) or isinstance(lb,int):
        hlb = np.array([-lb for ii in range(rets_mat.shape[1])])
    else:
        hlb = np.array([-x for x in lb])
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
        if isinstance(ub,float):
            h = np.array([ub for ii in range(rets_mat.shape[1])]+[-lb for ii in range(rets_mat.shape[1])] + sector_bnd)
        else:
            h = np.array([ y for y in ub]+[-x for x in lb] + sector_bnd)
            


    A = np.array([1.0 for ii in range(rets_mat.shape[1])])
    b = np.array([1.0])

    #lsq_sol =  solve_qp(P, q, G, h, A, b, solver="cvxopt")
    #ecos,osqp,quadprog
    lsq_sol =  solve_qp(P, q, G, h, A, b, solver="quadprog")
    sol_d = {rets_mat.columns[ii]: lsq_sol[ii] for ii in range(len(rets_mat.columns))}
    itms = list(sol_d.items())
    itms.sort(key = lambda x:x[1], reverse = True)
    print(itms[:60])
    print("#@#@")
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

#def wrap_rebalancing(res1,Ticker2Sector,d_weights_sector,D_tickers,bnd):
def wrap_rebalancing(res1,Ticker2Sector,D_tickers,bnd):
    """
    D_sector_w = rebalancing_results(dict(res1),Ticker2Sector)
    print(D_sector_w)
    print(d_weights_sector)
    #D_fact,D_no_w = updating_rebalancing_weights(d_weights_sector,D_sector_w)
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
    """
    aprx1 = create_aprx1(res1, D_tickers)
    #return aprx1,D_w_after
    return aprx1,res1

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
def compute_lsq_from_tickers():
    return

def match_dates(D_tickers_orig,df_tar,target_ret, match_d, d2h,forbidden,sector_bounds,num_of_tickers,ub,lb=0):
    #print(target_ret.index.values[-10:])
    
    print("forbidden",forbidden)
    keys_list = list(match_d.keys())
    keys_list.sort()
    print(keys_list)
   
    print(df_tar.index.values[:10])
    print((df_tar.index.values[3]), date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    print((df_tar.index.values[3]) > date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    # df_tar.index = pd.to_datetime(df_tar.index)

    df_tar = df_tar.loc[date_parser(keys_list[0]).strftime("%Y-%m-%d"):]
    df_tar2 = df_tar.copy()
    print("*" * 50)
    print(keys_list[0], date_parser(keys_list[0]).strftime("%Y-%m-%d"))
    
    print(df_tar.index.values[:10])
    
    
    #wts_base = d2h[date_parser(keys_list[0]).strftime("%Y-%m-%d")]
    #weight_col = [c for c in wts_base.columns if c.lower().find("weight") > -1][0]
    #ticker_col = [c for c in wts_base.columns if c.lower().find("ticker") > -1][0]
    
    print("keys_list",keys_list)
    D_fin = {}
    for ii in range(len(keys_list)):
        print("match dates ", ii, keys_list[ii])
        #univ_cof = pd.read_csv(os.path.join(".
        
        print("=$")
        print(os.listdir(os.path.join("..","data","holdings","IVV")))
        k = keys_list[ii]
        f_cand = os.path.join("..","data","holdings","IVV",'IVV_holdings_%s_f.csv'%k)
        if os.path.isfile(f_cand):
            print(f_cand)
            
            etf_holdings_tickers = list(pd.read_csv(f_cand)["Ticker"])
            etf_holdings_tickers_f = [x for x in etf_holdings_tickers if not x in  forbidden]
            tickers_weight_d = pd.read_csv(f_cand).set_index("Ticker")["Weight (%)"].to_dict()
            relative_weight = pd.read_csv(f_cand).set_index("Ticker")["Weight (%)"].loc[etf_holdings_tickers_f][:num_of_tickers].sum()*0.01
            const_w_c = 1./relative_weight 
            tickers_weight_d = {x:0.01*tickers_weight_d[x] for x in tickers_weight_d.keys() if isinstance(tickers_weight_d[x],float)}
            print(len(tickers_weight_d.keys()))
            etf_holdings_tickers = [x for x in etf_holdings_tickers if x in tickers_weight_d.keys()]
            
        else:
            f_cands = os.listdir(os.path.join("..","data","holdings","IVV"))
            f_cands = [x for x in f_cands if x.find("IVV_holdings")>-1]
            sol1 = max([x for x in f_cands if x < 'IVV_holdings_%s_f.csv'%k])
            
            etf_holdings_tickers = list(pd.read_csv(os.path.join("..","data","holdings","IVV",sol1))["Ticker"])
            etf_holdings_tickers_f = [x for x in etf_holdings_tickers if not x in  forbidden]
            f_cand = os.path.join("..","data","holdings","IVV",sol1)
            relative_weight = pd.read_csv(f_cand).set_index("Ticker")["Weight (%)"].loc[etf_holdings_tickers_f][:num_of_tickers].sum()*0.01
            const_w_c = 1./relative_weight 
            tickers_weight_d = pd.read_csv(f_cand).set_index("Ticker")["Weight (%)"].to_dict()
            tickers_weight_d = {x:0.01*tickers_weight_d[x] for x in tickers_weight_d.keys() if isinstance(tickers_weight_d[x],float)}
            print(len(tickers_weight_d.keys()))
            etf_holdings_tickers = [x for x in etf_holdings_tickers if x in tickers_weight_d.keys()]
            
        
        D_tickers = D_tickers_orig.copy()
        D_tickers = {x:D_tickers[x] for x in D_tickers.keys() if x in etf_holdings_tickers}
        dt = date_parser(k).strftime("%Y-%m-%d")
        #print(d2h[dt])
        years_bef = 2
        year_before = str(int(dt.split("-")[0])-years_bef)
        start_dt = dt#year_before + "-"+dt.split("-")[1]+"-"+dt.split("-")[2]
        dt_year = int(dt.split("-")[0])
        start_dt = str(year_before) +"-" + dt[5:]
        print("&^&^")
        print("start_dt %s"%start_dt)
        print("end_dt %s"%dt)
        #start_dt = date_parser(match_d[keys_list[ii - 1]]).strftime("%Y-%m-%d")
        end_dt  = dt#date_parser(match_d[keys_list[ii+2]]).strftime("%Y-%m-%d")
        D_tickers2 = {x: D_tickers[x].loc[start_dt:end_dt] for x in D_tickers.keys() if not x in forbidden}
        test_key = list(D_tickers2.keys())[0]
        print(start_dt,end_dt)
        print(test_key)
        print("target_ret3")
        print(target_ret.index.values[:5])
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
        llb = [max(tickers_weight_d[x]*0.7*const_w_c,lb) for x in D3.keys()]
        uub = [ min(tickers_weight_d[x]*1.5*const_w_c,ub) for x in D3.keys()]
        res_ds = lsq_with_constraints(D3, target_ret.loc[start_dt:end_dt]["return"], lb, ub, start_dt, end_dt,
                                      Sector2Tickers, sector_bounds)
        itms = list(res_ds.items())
        itms.sort(key = lambda x:x[1], reverse = True)
        itms = itms[: min(num_of_tickers,len(itms))]
        itms = [x[0] for x in itms]
        print(len(itms))
        itms = etf_holdings_tickers[:num_of_tickers]
        
        D3 = {x:D3[x] for x in D3.keys() if x in itms and x!="USD"}
        D3_cp = D3.copy()
        llb = [max(tickers_weight_d[x]*0.95*const_w_c,lb) for x in D3.keys()]
        uub = [ min(tickers_weight_d[x]*1.3*const_w_c,ub) for x in D3.keys()]
        #llb = [max(tickers_weight_d[x]*0.99,lb) for x in D3.keys()]
        #uub = [ min(tickers_weight_d[x]*1.1,ub) for x in D3.keys()]
        #llb = [max(lb,lb) for x in D3.keys()]
        #uub = [ min(ub,ub) for x in D3.keys()]
        kys = list(D3.keys())
        for x in range(len(kys)):
            print(kys[x],llb[x],uub[x])
            if llb[x] > uub[x]:
                print("?"*50)
                llb[x] = max(0.0,uub[x] - 0.01)
                uub[x] = max(0.0,uub[x])
            else :
                llb[x] = max(0,llb[x])
                uub[x] = max(0,uub[x])
        #llb = [lb for x in D3.keys()]
        #uub = [ub for x in D3.keys()]
        res_ds = lsq_with_constraints(D3, target_ret.loc[start_dt:end_dt]["return"], llb, uub, start_dt, end_dt,
                                      Sector2Tickers, sector_bounds)
        print(len([x for x in D3.keys()]))
        
        tickers_weight_x = pd.read_csv(f_cand).set_index("Ticker")["Weight (%)"].to_dict()
        tickers_weight_x = {x:0.01*tickers_weight_x[x] for x in tickers_weight_x.keys() if isinstance(tickers_weight_x[x],float) and x in D_tickers_orig.keys() and 'return' in D_tickers_orig[x].columns}
        tickers_weight_x = {x:tickers_weight_x[x] for x in tickers_weight_x.keys() if x!="USD"}
        tickers_weight_x = {x:tickers_weight_x[x] for x in tickers_weight_x.keys() if re.findall('[A-Z]+',x) and re.findall('[A-Z]+',x)[0] == x}
        sm11 =  sum(tickers_weight_x.values())
        tickers_weight_x = {x: tickers_weight_x[x]/sm11 for x in tickers_weight_x.keys()}
        res_ds = tickers_weight_x
        
        # aprx_ns = create_aprx1(res_ds, D_tickers)
        #aprx_ns, D_w_after = wrap_rebalancing(res_ds, Ticker2Sector, d_weights_sector, D_tickers,bnd=4)
        print(len([x for x in res_ds.keys() if res_ds[x]>0]))
        print("total weights before",sum(res_ds.values()))
        #aprx_ns, D_w_after = wrap_rebalancing(res_ds, Ticker2Sector, D_tickers,bnd=4)
        print("total weights after",sum(res_ds.values()))
        
        d1 = res_ds#D_w_after
        if ii < len(keys_list)-1:
            ks1 = match_d[keys_list[ii + 1]]
            dts1 = date_parser(ks1).strftime("%Y-%m-%d")
            print("dts1 %s"%dts1)
           
            D_t_rets = {k: D_tickers_orig[k]["return"] for k in d1.keys()}
            print(D_t_rets.keys())
            test_key = list(D_t_rets.keys())[0]
            print(D_t_rets[test_key].shape)
            df_tar1 = target_ret.loc[dt:dts1]
            print("tar_ret shape", df_tar1.shape)
            rets_mat = pd.DataFrame(D_t_rets)
            rets_mat = rets_mat.loc[df_tar1.index.values]
            print("rets_mat shape ",rets_mat.shape)
            rets_mat_cp = rets_mat.copy()
            for jj in d1.keys():
                # print(k)
                tmp_t_ret = rets_mat[jj].copy()
                #tmp_t_ret.iloc[0] = 0.0
                tmp1 = (1. + tmp_t_ret).cumprod()
                #tmp1 = (1. + rets_mat[jj]).cumprod()
                print(jj)
                print(tmp1.tail())
                
                if False: #ii == 1:
                    df_tar[jj].loc[:dts1] = d1[jj]
                else:
                    rets_mat_cp[jj] =  d1[jj]*tmp1
                    df_tar[jj].loc[dt:dts1] = d1[jj]
            D_fin[(dt,dts1)] = rets_mat_cp
            if ii == len(keys_list)-2:
                print(dt,dts1)
                print(df_tar["MSFT"].loc[dt:dts1])
                print(df_tar[["AAPL","MSFT","XOM","BAC"]].head())
                print(keys_list)
                
        else:
            df_tar1 = target_ret.loc[dt:]
            D_t_rets = {k: D_tickers_orig[k]["return"] for k in d1.keys()}
            print(D_t_rets.keys())
            test_key = list(D_t_rets.keys())[0]
            print(D_t_rets[test_key].shape)
            print(df_tar1.index.values[-10:])
            
            df_tar1 = target_ret.loc[dt:]
            print("tar_ret shape", df_tar1.shape)
            print(df_tar1.index.values[-10:])
            rets_mat = pd.DataFrame(D_t_rets)
            print("dt ",dt," rets_mat.index.values[-1]",rets_mat.index.values[-1])
            
            rets_mat = rets_mat.loc[df_tar1.index.values]
            print("rets_mat shape ",rets_mat.shape)
            
            rets_mat_cp = rets_mat.copy()
            for jj in d1.keys():
                tmp_t_ret = rets_mat[jj].copy()
                print(jj)
                print("df_tar1.index.values[-10:]",df_tar1.index.values[-10:])
                print(tmp_t_ret.shape)
                print(">"*30)
                tmp_t_ret.iloc[0] = 0.0
                tmp1 = (1. + tmp_t_ret).cumprod()
                
                print(tmp1.tail())
                
                df_tar[jj].loc[dt:] = d1[jj]
                rets_mat_cp[jj] =  d1[jj]*tmp1
            D_fin[(dt,"end")] = rets_mat_cp
                
    
    """
    k = keys_list[-1]
    print(k)
    dt = date_parser(k).strftime("%Y-%m-%d")
    weights = d2h[dt].set_index(ticker_col)[weight_col]
    eligible_tickers = list(df_tar.columns)
    eligible = [c for c in weights.index.values if c in eligible_tickers]
    weights = weights.loc[eligible]
    sm1 = weights.sum()
    weights *= 1. / sm1
    d1 = weights.to_dict()
    print(dt)
    for k in d1.keys():
        df_tar[k].loc[dt:] = d1[k]
    """
    print("="*50)
    print(df_tar[["AAPL","JNJ","JPM"]].tail(),df_tar[["AAPL","JNJ","JPM"]].head())
    
    
    return df_tar,D_fin

def compute_mat_ret(mat):
    total = mat.sum(axis=1)
    print(total.head())
    f_day = total.iloc[0] -1
    total = total.pct_change().fillna(0.0)
    total.iloc[0] = f_day
    return total
def compute_return_intr(D_fin):
    #aprox = compute_return(cpn,tickers_pv, start_dt,list(match_d.keys()))
    for kk in  D_fin.keys():
        print(kk)
        tmp2 = compute_mat_ret(D_fin[kk])
        print(tmp2.head())
        print("="*40)
        print(tmp2.tail())
        print("$"*40)
    s1 =  pd.concat([compute_mat_ret(x) for x in D_fin.values()])#.drop_duplicates(keep ="first")
    
    s1 = s1[~s1.index.duplicated(keep='first')]
    return pd.DataFrame(s1,columns=["return"])
    
    
def wrapper_strategy(PriceVolume_dr,index_df,index_holdings_path,match_d,constraints,start_dt,end_dt,sector_mapping):
    df_tar = create_universe_zero_df(PriceVolume_dr,index_df).loc[start_dt:end_dt]
    d2h = dates_2_holdings_dict(index_holdings_path)
    print(index_df.head(),index_df.tail())
    
    print(start_dt,end_dt)
    #date_parser
    
    match_d = {x:match_d[x] for x in match_d.keys() if date_parser(x).strftime("%Y-%m-%d")>=start_dt and date_parser(x).strftime("%Y-%m-%d")<=end_dt}
    print(start_dt,end_dt)
    print(match_d)
    #zzzz
    
    
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
    print("tickers_pv num of elements %d"%(len(tickers_pv.keys())))
    #sdasdda
    index_df = index_df[:end_dt]
    df_tar,D_fin = match_dates(tickers_pv, df_tar,index_df, match_d, d2h, forbidden, sector_bounds, num_of_tickers, ub, lb=0)
    
    print(D_fin.keys())
    print("=*=")
    #xxxx
    d_aprox2 = compute_return_intr(D_fin)
    print(d_aprox2.head())
    print("="*50)
    print(d_aprox2.tail())
    print("="*50)
    print(d_aprox2[50:70])
    print(d_aprox2[-70:-50])
    
    #match_dates(tickers_pv,df_tar, match_d, d2h, constraints["forbiden_tickers"],constraints["sectors"],constraints["num_of_tickers"],constraints["upper_bound"],sector_mapping)
    universe = list(df_tar.columns)
    print("*"*20)
    print(df_tar["AAPL"].tail())
    
    close_col = [c for c in index_df.columns if c.lower().find("close") > -1]
    if len([c for c in close_col if c.lower().find("adj") > -1]) > 0:
        close_col = [c for c in close_col if c.lower().find("adj") > -1][0]
    else:
        close_col = close_col[0]

    #filter
    #df_tar = filter_out_forbiden_tickers(df_tar, constraints["forbiden_tickers"])
    cpn = df_tar.copy()
    #L = list(D_fin.values())
    #L = [L[0]] + [x[1:] for x in L[1:]]
    #cpn = pd.concat(L,axis=0).drop_duplicates(keep="first").fillna(0.0)
    #aprox = compute_return(cpn,tickers_pv, start_dt,list(match_d.keys()))
    aprox = d_aprox2.loc[start_dt:end_dt]
    #aprox = compute_return(cpn,tickers_pv, start_dt,list(match_d.keys()))
    print("start_dt %s"%(start_dt))
    aprox["benchmark_index_return"] = index_df.loc[start_dt:end_dt]["return"]
    aprox["Comulative_ret"] = (1. + aprox["return"]).cumprod()
    aprox["benchmark_index_comulative_ret"] = (1. + aprox["benchmark_index_return"]).cumprod()
    print(">"*30)
    print(aprox["return"][:10])
    print(">"*30)
    print(aprox["benchmark_index_return"][:10])
    ####
    print(">"*30)
    print(aprox["return"][60:70])
    print(">"*30)
    print(aprox["benchmark_index_return"][60:70])
    return aprox[:],df_tar

GICS = pd.read_csv(os.path.join("..","data","GICS","GICS_sector_SP500.csv"))
Ticker2Sector = GICS.set_index("Ticker")["Sector GICS"].to_dict()
SectorMapping = {}
for k in Ticker2Sector.keys():
    SectorMapping[Ticker2Sector[k]] = SectorMapping.get(Ticker2Sector[k],[])+[k]
add_h = SectorMapping.pop("Health")
SectorMapping["Health Care"] += add_h

if __name__ == "__main__":
    start = time.time()
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
    year = 2015
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
    input_file = os.path.join("..","example","input_files","InputExample2022.txt")
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
    out_dir = os.path.join("..","..","lsq_test_res_2022")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    generate_basic_stats(aprox, out_dir, "temp")
    print("Total time is %0.3f"%(time.time()-start))
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

