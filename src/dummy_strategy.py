import pandas as pd
from dates_matching import  dates_2_holdings_dict,matching_dates_d
#dates_matching
def dummy_strat_index(index_returns,rebalncing_dates,index_holdings_path,constraints):
    rebalncing_dates_dict = {rebalncing_dates[ii] : rebalncing_dates[ii-1] for ii in range(1,len(rebalncing_dates))}
    d2h = dates_2_holdings_dict(index_holdings_path)
    matching_d = matching_dates_d(list(rebalncing_dates_dict.keys()),list(d2h.keys()))
    weights_d = {d: d2h[matching_d[d]] for d in rebalncing_dates_dict.keys()}




