import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from scipy.optimize import minimize, Bounds
from scipy.optimize import minimize_scalar
from scipy import special
import sympy as sp
import math


# ----TEMPURATURE----
# general
general_T = 293
# LL
#T = 310
# Mittal
#T = 300
T = 298


# ----KUHN LENGTH----
# Kuhn length (Angstroms) (sometimes written without a subscript)
l_k = 8
# bond length
b = 3.8
# Bjerrum Kuhn Length (7.12 at 20C)
l_b_20 = 7.12
l_b = l_b_20 * (general_T / T)

#salt = 150

# ----SALT----
# convert to mol/L, cancel mol, cancel/convert liters to cm^3, cancel/convert cm^3 to A^-3
def mM_to_A(mM):
    return mM * 10**(-3) * 6.022*10**(23) * (1/1000) * 1 / (10**8)**3

def kl_func(salt): 
    return np.sqrt(8 * math.pi * l_b * mM_to_A(salt)) * b

#----OMEGA 3----
w3 = .2


#----Constants----
amino_acid_data = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "E": -1,
    "Q": 0,
    "G": 0,
    "H": .5,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
    "B": 2,
    "Z": -2
}

pKa_values = {
    "R": 12.3,
    "D": 3.5,
    "C": 6.8,
    "E": 4.2,
    "H": 6.6,
    "K": 10.5,
    "Y": 10.3,
    "B": 7.7,
    "Z": 3.3
}

s_values = {
    "R": 1,
    "D": -1,
    "C": -1,
    "E": -1,
    "H": 1,
    "K": 1,
    "Y": -1,
    "B": 1,
    "Z": -1
}

#----Functions----
def adjust_pH(pH):
    for key in pKa_values:
        if key in amino_acid_data:
            s = s_values[key]
            amino_acid_data[key] = convert_charge_pH(pH, key, s)

def convert_charge_pH(pH, key, s):
    pKa = pKa_values[key]
    return s / (1 + 10**(s*(pH - pKa)))

def get_x(Ree, N):
    return Ree**2 / (N * b * l_k)

# * .1 converts to nanometers 
def get_Ree(x, N):
    return np.sqrt(x * (N * b * l_k)) * .1

# * .1 converts to nanometers
def get_Rg(x, N):
    return np.sqrt(x * N * b * l_k / 6) * .1

def process_seq(seq, n, c, idp):
    new_seq = seq

    n_aa = new_seq[0]
    c_aa = new_seq[-1]
    
    if n or idp:
        if n_aa == "R" or n_aa == "K":   new_seq = "B" + new_seq[1:]
        elif n_aa == "D" or n_aa == "E": new_seq = "A" + new_seq[1:]
        else:                            new_seq = "R" + new_seq[1:]
            
    if c or idp:
        if c_aa == "R" or c_aa == "K":   new_seq = new_seq[:-1] + "A"
        elif c_aa == "D" or c_aa == "E": new_seq = new_seq[:-1] + "Z"
        else:                            new_seq = new_seq[:-1] + "D"

    return new_seq

def get_charge(m, n, seq):
    aam = seq[m-1]
    aan = seq[n-1]
    
    qm = amino_acid_data.get(aam)
    qn = amino_acid_data.get(aan)
    
    return qm * qn

def set_vars(data, i):
    name = data.iloc[i, 0]

    raw_seq = data.iloc[i, 2]
    N   = int(data.iloc[i, 3])
    nterm = data.iloc[i, 4]
    cterm = data.iloc[i, 5]
    nandc = data.iloc[i, 6]
    
    x = data.iloc[i, 9]
    w2  = data.iloc[i, 10]
    if w2 != "None":
        w2  = float(w2)
    
    seq = process_seq(raw_seq, nterm, cterm, nandc)
    
    return N, w2, seq, x, name

def calc_x_w_load(N, w2, seq, seed, O_term, B_term, salt, pH):
    adjust_pH(pH)
    print("salt:", salt, "kappa:", kl_func(salt))
    mn_array = mnArray_Q_prime(N, seq)
    bounds = Bounds(.01, 10, keep_feasible=False)
    result = minimize(function_to_solve, seed, method="Nelder-Mead", args=(N, w2, seq, mn_array, O_term, B_term, salt), bounds=bounds)
    x = result.x[0]
    return x, get_Ree(x, N), get_Rg(x, N)

# add 1 to end of sum, python is non-inclusive
def Omega(N, w2):
    result = 0.0
    for m in range(2, N + 1):
        for n in range(1, m):
            result += w2 * ((m - n) ** (-0.5))
    return 1/N * result

def mn_Omega(N):
    result = 0.0
    for m in range(2, N + 1):
        for n in range(1, m):
            result += (m - n) ** (-0.5)
    return 1/N * result

def B(N):
    result = 0.0
    for p in range(3, N+1):
        for m in range(2, p):
            for n in range(1, m):
                result += (p - n)/(((p-m)*(m-n))**(3/2))
    return 1/N * result

# output scd to test
def Q(N, seq):
    result = 0.0
    for m in range(2, N+1):
        for n in range(1, m):
            result += get_charge(m, n, seq) * ((m - n) ** (0.5))
    output = 1/N * result
    #print(output)
    return output

def Q_prime(N, seq, x):
    result = 0.0
    for m in range(2, N+1):
        for n in range(1, m):
            result += get_charge(m, n, seq) * ((m-n)**2) * A_prime(m, n, x)
    output = 1/N * result
    return output

def mn_Q_prime(N, seq, x, mn_array, salt):
    result = 0.0
    i = 0
    for m in range(2, N+1):
        for n in range(1, m):
            result += mn_array[i] * A_prime(m, n, x, salt)
            i += 1
    output = 1/N * result
    return output

def mnArray_Q_prime(N, seq):
    result = []
    for m in range(2, N+1):
        for n in range(1, m):
            result.append(get_charge(m, n, seq) * ((m-n)**2))
    return result

def A_prime(m, n, x, salt):
    term1 = 1/2 * (6*math.pi/x)**(1/2) * (1/(m-n)**(3/2))
    term2 = kl_func(salt) * (math.pi/2) * (1/(m-n))
    term3 = special.erfcx(np.sqrt(kl_func(salt)**2 * x * (m-n) / 6))
    return term1 - term2 * term3

def free_energy(x, N, w2, seq, mn_array, O_term, B_term, salt):
    # beta F(x) = 3/2 (x-ln(x))
    #             + (3/(2pi))**(2/3)   * Omega        * 1/x**(3/2)
    #             + w_3 (3/(2 pi))**3  * B/2          * 1/(x**3)
    #             + l_b / l_k          * Q*sqrt(6/pi) * 1/x**(1/2)

    #Q_term = Q(N, seq)
    Q_term = mn_Q_prime(N, seq, x, mn_array, salt)
    # Define the equation
    eq = ( (
                   3/2                                            * (x - np.log(x))
                 + (3/(2*math.pi))**(3/2)              * O_term   * (1/(x**(3/2)))
                 + (w3*(3/(2*math.pi))**(3))/2           * B_term   * (1/(x**3))
                 + (l_b / b) * 2/math.pi  * Q_term
                 ))

    return eq

def function_to_solve(argument, N, w2, seq, mn_array, O_term, B_term, salt):
    """function, to be solved."""

    x = argument

    sol = free_energy(x, N, w2, seq, mn_array, O_term, B_term, salt)
    return sol


def Q_prime_derv(N, seq, x):
    result = 0.0
    for m in range(2, N+1):
        for n in range(1, m):
            result += get_charge(m, n, seq) * ((m-n)**2) * A_prime_derv(m, n, x)
    output = 1/N * result
    return output

def A_prime_derv(m, n, x):
    term1 = (np.sqrt(math.pi)/4) * (6/x)**(3/2) * (1/(m-n)**(3/2))
    term2 = kl**2 * (np.sqrt(math.pi)/2) * (6/x)**(1/2) * (1/(m-n)**(1/2))
    term3 = kl**3 * (math.pi/2) * special.erfcx(np.sqrt(kl**2 * x * (m-n) / 6))
    return (-1/6) * (term1 - term2 + term3)

def solve_for_w2_eq(w2, N, x, seq, mn_O_term, B_term, Q_term):
    O_term = mn_O_term * w2

    # Derivative of the free energy equation
    eq = ( 
                   3/2                                                 * (x - 1)/x
                 + (-9 * np.sqrt(3/2)/(4*(math.pi)**(3/2))) * O_term   * (1/(x**(5/2)))
                 + (-w3 * 81) / (16 * (math.pi)**(3))       * B_term   * (1/(x**4))
                 + (l_b / b) * 2/math.pi                    * Q_term 
                 )    
    
    return eq

def solve_for_w2(N, seq, x):

    B_term = B(N, seq)
    Q_term = Q_prime_derv(N, seq, x)
    mn_O_term = mn_Omega(N)

    # Define the variable
    w2 = sp.Symbol('w2')

    # Define the equation eq(x, y) = 0
    equation = solve_for_w2_eq(w2, N, x, seq, mn_O_term, B_term, Q_term)

    # Solve the equation for the given value of y
    solutions = sp.solve(equation, w2)
    return solutions[0]
