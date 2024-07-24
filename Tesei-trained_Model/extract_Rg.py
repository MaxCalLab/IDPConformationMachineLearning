"""
Title:   Calculate R_g, R_ee and x using Tesei-trained PML model Omega_2 values
Author:  Lilianna Houston, Ghosh Lab
Date:    July 22nd 2024
Purpose: This code calculates the R_g, R_ee and x values of protein sequences 
         using omega_2 values extracted from the Tesei-trained ML model. 
Inputs:  CSV of protein sequences and omega_2 (w2) values.
Outputs: CSV of protein sequences with R_g, R_ee and x values.
"""

# Path to the CSV file containing protein sequences and omega_2 values
data_path = "exper_seqs_w2preds.csv"
# Specify sequence column
seq_column = 3
# Specify salt column
salt_column = 5
# Specify pH column
pH_column = 6
# Specify omega_2 column
w2_column = 10

# Import packages
import theory_functions # Custom module with all theory functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from scipy.optimize import minimize, Bounds
from scipy.optimize import minimize_scalar
from scipy import special
import sympy as sp
import math
import time
import sys

# Load the data from the CSV file
data = pd.read_csv(data_path)

# Initialize lists to store results
xs = np.zeros(len(data))
Rs = np.zeros(len(data))
Rgs = np.zeros(len(data))

for i in range(0, 3):
    o_seq = data.iloc[i, seq_column]
    seq = theory_functions.process_seq(o_seq, False, False, True)
    N = len(seq)
    salt = data.iloc[i, salt_column] 
    pH = data.iloc[i, pH_column]
    w2 = data.iloc[i, w2_column]

    # Load pre-calculated Omega and B values
    OBlist = []
    with open('OBfmt_5-1500.npy', 'rb') as f:
        OBlist.append( np.load(f) )
    OBfmt = OBlist[0]

    # Get the Omega and B values corresponding to the sequence length N
    index = np.where(OBfmt[:,0]==N)[0][0]
    Omega, B = OBfmt[index,1:]

    O_term = Omega*w2
    B_term = B

    # Calculate x, R_ee, and R_g using the theoretical model
    x, Ree, Rg = theory_functions.calc_x_w_load(N, w2, seq, .1, O_term, B_term, salt, pH)
    xs[i]  = (x)
    Rs[i]  = (Ree)
    Rgs[i] = (Rg)

# Add the values to the DataFrame and save it to a CSV file
data["x_pred"]   = xs
data["Ree_pred"] = Rs
data["Rg_pred"]  = Rgs
data.to_csv('exper_Rg_preds.csv', index=False)