import numpy as np
import os,sys,re
import pandas as pd
import matplotlib.pyplot as plt
from basic_reader import max_dd
from risk_L2 import Risk
def generate_basic_stats(df,output_dir,name):
    aprox2[["Comulative_ret", "benchmark_comulative_ret"]].plot()
    plt.savefig(os.path.join(output_dir,'benchmarkVstracker.png'))

if __name__ == "__main__":
    print("In")