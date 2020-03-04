import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import scipy.special as sp
from scipy.linalg import lstsq
from sys import argv, exit

#
if (len(argv)!=2):
    print("Invalid Data File. Expected input: python3 ee18b104_asgn3.py <data_file_name> ('fitting.dat').\n Exiting")
    exit()
try:
    data = pl.loadtxt(argv[1])
except Exception as z:
    print(z)
def g(t, A, B):		# Computes and returns the function A*J(2, t) + B*t
	y = A*sp.jn(2, t) + B*t
	return y
def M_p(t, A, B):	# Computes A*J(2, t) + B*t as a matrix product, using matrices makes it easier for least sq regression
	p = [A, B]
	return np.matmul(M, p)
def data_vis(i):	# Generates subplots for visualising input data
	fig_vis, axs = plt.subplots(2)
	axs[0].plot(t, true_Mp)		
	axs[0].set_title("Output of function M_p()")
	axs[1].errorbar(t[::5], f[i][::5], sigma[i], fmt = 'ro')
	axs[1].plot(t, true_value)
	axs[1].legend(["Output of function g()", "Errorbar for column "+ str(i+1)])
	axs[1].set_title("Errorbar for column " + str(i+1)+" data")
	fig_vis.tight_layout()
	err = np.sum((true_Mp - true_value)**2)
	print("Error between M_p() and g() (SANITY CHECK) = ", err)
	return 0
#
def plot_soln(i):	# Generates contour plot of MSE and plots of solution 
	fig, axs = plt.subplots(2)
	contour_err = axs[0].contour(A, B, mse[i])
	axs[0].clabel(contour_err, inline = 1, fontsize = 10)
	axs[0].plot(soln[i][0], soln[i][1], 'ro')
	axs[1].plot(t, M_p(t, soln[i][0], soln[i][1]))
	axs[1].plot(t, true_Mp)
	axs[1].plot(t, f[i])
	axs[1].legend(["solution", "true value", "Raw Data"])
	fig.suptitle("Contours and Solution")
	print("Error in the solution w.r.t true value = ", soln_err[i])
	return 0
#
t = data[:, 0]	
data = data[:, 1:]
f = np.zeros((data.shape[1], data.shape[0]))
sigma = np.logspace(-1, -3, 9)
x = sp.jn(2, t)
M = pl.c_[x, t]
true_A = 1.05; true_B = -0.105;		# Known true values of A and B
true_value = g(t, true_A, true_B)
true_Mp = M_p(t, true_A, true_B)
A = [i/10 for i in range(21)]	# According to the hypothesis, 'A' belongs to the range [0, 2]; step size of 0.1 is taken for visualising contours
B = [-1*j/100 for j in range(21)] # 'B' belongs to [-0.2, 0]
#
soln_err = np.zeros(data.shape[1])
soln_err -= np.ones_like(soln_err)	
err_A = np.zeros(data.shape[1])
err_A -= np.ones_like(err_A)
err_B = np.zeros(data.shape[1])
err_B -= np.ones_like(err_B)
# Write absurd value into soln_err initially- this is done so that we may identify situations where the solution error is not computed easily 
soln = np.zeros((data.shape[1], 2))
mse = np.zeros((data.shape[1], len(A), len(B)))
for i in range(data.shape[1]):
	f[i] = data[:, i]
	for j in range(len(A)):
		for k in range(len(B)):
			mse[i, j, k] += np.sum((f[i] - M_p(t, A[j], B[k]))**2)/101  # run A and B through the 'expected' range and visulaise the MSE obtained
	soln[i] = lstsq(M, f[i])[0]
	soln_err[i] = np.sum((f[i] - M_p(t, soln[i][0], soln[i][1]))**2)/101  
	err_A[i] = abs(soln[i][0] - true_A)
	err_B[i] = abs(soln[i][1] - true_B)
# User interaction:
input_req = "Enter column of " + str(argv[1]) + " that has to be fit: "
i = int(input(input_req))
# Generate Plots
data_vis(i-1)
plot_soln(i-1)
# Plot error versus Stdev
fig_err, axs = plt.subplots(2)
axs[0].plot(sigma, soln_err); axs[0].plot(sigma, err_A, "go"); axs[0].plot(sigma, err_B, "rx")
# Plot Solution error, Error in A (with green 'o's) and Error in B (with red 'x's)
axs[0].legend(["Error in solution", "Error in A", "Error in B"])
axs[0].set_xlabel("Stdev of noise")
axs[0].set_ylabel("Solution error")
axs[0].set_title("Solution error v/s stdev- linear scale")
axs[1].loglog(sigma, soln_err); axs[1].loglog(sigma, err_A, "go"); axs[1].loglog(sigma, err_B, "rx")
axs[1].set_xlabel("Stdev of noise")
axs[1].set_ylabel("Solution error")
axs[1].set_title("Solution error v/s stdev- loglog scale" )
fig_err.tight_layout(),
plt.show()
# Print solution- note that the obtained values of A and B are close to the theoretical values 
print("Optimal value of A and B = ", soln[i])
