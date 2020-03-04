import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import lstsq
from math import pi
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
# Functions to generate function sequences
def gen_exp(x):
    return np.exp(x)
def gen_ccos(x):
    return np.cos(np.cos(x))
# u and v are the integrands of the Fourier Series Analysis equations
def u_exp(x, n):
    exp_x = gen_exp(x)
    return exp_x*np.cos(x*n)
def v_exp(x, n):
    exp_x = gen_exp(x)
    return exp_x*np.sin(x*n)
def u_ccos(x, n):
    ccos_x = gen_ccos(x)
    return ccos_x*np.cos(x*n)
def v_ccos(x, n):
    ccos_x = gen_ccos(x)
    return ccos_x*np.sin(x*n)
# Function to regenerate function sequence from the Fourier Series Coefficients obtained by Numerical Integration
def regen(fn_type):
    if(fn_type == "exp"):
        est = np.ones(x.shape)*coeff_exp[0]
        for n in range(1, len(k)):
            est += coeff_exp[2*n -1]*np.cos(k[n]*x) + coeff_exp[2*n]*np.sin(k[n]*x)
    else:       # only two possibilities here
        est = np.ones(x.shape)*coeff_ccos[0]
        for n in range(1, len(k)):
            est += coeff_ccos[2*n -1]*np.cos(k[n]*x) + coeff_ccos[2*n]*np.sin(k[n]*x)
    return est
# Function to plot FS Coefficients in semilogy and loglog scales
def plot_coeff(coeff, title):
    fig, axs = plt.subplots(2)
    axs[0].semilogy(k[1::], abs(coeff[2::2]), "ro")
    axs[0].semilogy(k[1::], abs(coeff[1::2]), "bx")
    axs[0].semilogy(k[0], abs(coeff[0]), "bx")
    axs[0].set_title(title[0])
    axs[0].grid()
    axs[0].legend(["b_k", "a_k", "a_0"])
    axs[1].loglog(k[1::], abs(coeff[2::2]), "ro")
    axs[1].loglog(k[1::], abs(coeff[1::2]), "bx")
    axs[1].loglog(k[0], abs(coeff[0]), "bx")
    axs[1].set_title(title[1])
    axs[1].grid()
    axs[1].legend(["b_k", "a_k", "a_0"])
    fig.tight_layout()
    return fig  
# Function to plot regenerated function sequences
def plot_regen(est1, est2, title):
    fig, axs = plt.subplots(2)
    axs[0].semilogy(x, est1)
    axs[0].semilogy(x1, exp1)
    axs[0].set_title("Regenerated exp(x)- " + title)
    axs[0].grid()
    axs[0].set_xlabel("x ->")
    axs[0].set_ylabel("log(y)")
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('(%g/3)$\pi$'))     # Use ticker frequency = pi instead of integers
    axs[0].xaxis.set_major_locator(MultipleLocator(base=2.0))
    axs[1].plot(x, est2)
    axs[1].plot(x1, ccos1)
    axs[1].set_title("Regenerated cos(cos(x))- " + title)
    axs[1].grid()
    axs[1].set_xlabel("x ->")
    axs[1].set_ylabel("y")
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('(%g/3)$\pi$'))
    axs[1].xaxis.set_major_locator(MultipleLocator(base=2.0))
    fig.tight_layout()
    return fig
#
x = np.linspace(0, 2*pi, 401)
x = x[:-1]
k = np.linspace(0, 25, 26)
coeff_exp = np.zeros(51); coeff_ccos = np.zeros(51)
coeff_exp[0] = quad(u_exp, 0, 2*pi, args = (k[0]))[0]/(2*pi)    # a_0 = Cycle Average of the function
# Generate FS Coefficients by Numeric Integration
for n in range(1, len(k)):
    coeff_exp[2*n - 1] = quad(u_exp, 0, 2*pi, args = (k[n]))[0]/(pi)
    coeff_exp[2*n] = quad(v_exp, 0, 2*pi, args = (k[n]))[0]/(pi)
coeff_ccos[0] = quad(u_ccos, 0, 2*pi, args = (k[0]))[0]/(2*pi)
for n in range(1, len(k)):
    coeff_ccos[2*n - 1] = quad(u_ccos, 0, 2*pi, args = (k[n]))[0]/(pi)
    coeff_ccos[2*n] = quad(v_ccos, 0, 2*pi, args = (k[n]))[0]/(pi)
est_exp = regen("exp")
est_ccos = regen("ccos")
# lstsq regression to find FS coefficients
b_exp = gen_exp(x)
b_ccos = gen_ccos(x)
A= np.zeros((400,51))
A[:,0]=1 #col 1 is all ones
for n in range(1,len(k)):
    A[:,2*n-1] = np.cos(k[n]*x) # cos(kx) column
    A[:,2*n] = np.sin(k[n]*x)   # sin(kx) column
c_exp=lstsq(A,b_exp)[0]
c_ccos = lstsq(A, b_ccos)[0]    
# Estimating error in coefficients obtained through different methods
mse_exp = np.sum((c_exp - coeff_exp)**2)
maxdev_exp = np.max(abs(c_exp - coeff_exp))
mse_ccos = np.sum((c_ccos - coeff_ccos)**2)
maxdev_ccos = np.max(abs(c_ccos - coeff_ccos))
print("Mean Square Errors in FS coefficients of exp(x) = ", mse_exp)
print("Mean Square Errors in FS coefficients of cos(cos(x)) = ", mse_ccos)
print("Max. Deviation in exp(x) = ", maxdev_exp)
print("Max. Deviation in cos(cos(x)) = ", maxdev_ccos)
# Regenerating thhe functions from FS coefficients obtained by lstsq
lst_est_exp = np.dot(A, c_exp)
lst_est_ccos = np.dot(A, c_ccos)
# Code for generating plots
x1 = np.linspace(-2*pi, 4*pi, 200)
exp1 = gen_exp(x1)
ccos1 = gen_ccos(x1)
# Plotting the FS coefficients- Q3
plot_coeff(coeff_exp, ["FS Coefficients of exp(x)- by FS integration (semilogy)", "FS Coefficients of exp(x)-  by FS integration (loglog)"])
plot_coeff(coeff_ccos, ["FS Coefficients of cos(cos(x))-  by FS integration (semilogy)", "FS Coefficients of cos(cos(x))-  by FS integration (loglog)"])
plot_coeff(c_exp, ["FS Coefficients of exp(x)- by lstsq (semilogy)", "FS Coefficients of exp(x)-  by lstsq (loglog)"])
plot_coeff(c_ccos, ["FS Coefficients of cos(cos(x))- by lstsq (semilogy)", "FS Coefficients of cos(cos((x))-  by lstsq (loglog)"])
# Plotting regenerated functions
plot_regen(est_exp, est_ccos, "FS integration")
plot_regen(lst_est_exp, lst_est_ccos, "lstsq")
plt.show()