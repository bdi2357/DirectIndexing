import numpy as np
import os,sys,re
import pandas as pd
import matplotlib.pyplot as plt
from basic_reader import max_dd
from risk_L2 import Risk
def generate_basic_stats(df,output_dir,name):
    stats = {}
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    df[[c for c in df.columns if c.lower().find("comulative")>-1]].plot()
    plt.savefig(os.path.join(output_dir,'benchmarkVstracker.png'))
    stats["maximum_abs_difference"] = max(abs(df["benchmark_index_comulative_ret"] - df["Comulative_ret"]))
    stats["anuallized_volatility"] = (df["benchmark_index_return"]-df["return"]).std() *(df.shape[0]**0.5)
    stats["max_dd"] = max(max_dd(1. + df["benchmark_index_comulative_ret"] - df["Comulative_ret"] ),max_dd(1. -df["benchmark_index_comulative_ret"] + df["Comulative_ret"] ))
    out_df = pd.DataFrame(stats.items(),columns= ["Stat","Value"])
    out_df.to_csv(os.path.join(output_dir,'stats.csv'))

if __name__ == "__main__":
    print("In")