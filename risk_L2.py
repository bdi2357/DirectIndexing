import numpy as np

def risk_L2(a,b, cnst = 1):
    return cnst*numpy.linalg.norm(a-b)

Risk = { "risk_func":risk_L2}