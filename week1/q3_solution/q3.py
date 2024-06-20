from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    return v * (t - ((1 - np.exp(-1*k*t)) / k))

    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    data = df.to_numpy()
    t = data[:,0]
    y = data[:,1]
    param , param_cov = curve_fit(func , t , y)
    v = param[0]
    k = param[1]
    # END TODO

    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    data = df.to_numpy()
    t = data[:,0]
    y = data[:,1]
    answer = func(t,v,k)
    plt.plot(t , y , '*' , color="blue",label="data")
    plt.plot(t , answer , '-' , color="red",label=f"fit: v={v}, k={k}")
    plt.legend()
    plt.savefig("fit_curve.png")
    # END TODO
