'''
		APPLIED PROGRAMMING LAB: EE2703 END SEMESTER EXAM
		JAN-MAY 2020 

		AUTHOR: ARJUN MENON VADAKKEVEEDU
		ROLL NUMBER: EE18B104
		DATE: 30th July 2020
'''

import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
from scipy.constants import epsilon_0 as eps0 
from numpy import format_float_scientific as roundf
import argparse as arg 
#
Lx = 10		#cm
Ly = 20
eps_r = 2	# dielectric constant
step = 0.1	# step size
M = int(Ly/step)
N = int(Lx/step)
h = 0.5*Ly
#h = 0.95*Ly
#As h approaches 1, a significant amount of E flows into the dielectric, resulting in higher Q and faster drop of potential
#
def solve_V(k, Nmax, delta = 1e-6):
	# Estimates V by solving Laplace equation iteratively through Finite Difference Methods
	phi = np.zeros((M, N))
	phi[0, :] = 1
	errors = []
	for num_iter in range(Nmax):
		oldphi = phi.copy()
		phi[1:k, 1:-1] = 0.25*(phi[0:k-1, 1:-1] + phi[2:k+1, 1:-1] + phi[1:k, 0:-2] + phi[1:k, 2:])
		phi[k+1:-1, 1:-1] = 0.25*(phi[k:-2, 1:-1] + phi[k+2:, 1:-1] + phi[k+1:-1, 0:-2] + phi[k+1:-1, 2:])
		phi[k, 1:-1] = (eps_r*phi[k+1, 1:-1] + phi[k-1, 1:-1])/(1+eps_r)
		errors.append(np.max(abs(phi - oldphi)))
		if errors[-1] < delta:
			break
	return phi, errors
#
def compute_grad(psi, step):
	# Computes 4-point gradient of psi and returns -grad(psi)
	grad_x = np.zeros((M-1, N-1))
	grad_y = grad_x.copy()
	grad_y[:, :] = ((psi[1:, 1:]+psi[1:, 0:-1]) - (psi[0:-1, 1:]+psi[0:-1, 0:-1]))/(2*step)
	grad_x[:, :] = ((psi[1:, 1:] + psi[0:-1, 1:])-(psi[1:, 0:-1] + psi[0:-1, 0:-1]))/(2*step)
	return grad_x*-1, grad_y*-1
#
def fit_err(k, slope, intercept):
	#input to curve_fit() for fitting log(err) v/s N to a linear map
	return k*slope + intercept
#
def eval_V_E(h, Nmax = 15000):
	# Wrapper function to evaluate E, V and error
	k = int((1 - h/Ly)*M)
	V, err = solve_V(k, Nmax)
	Ex, Ey = compute_grad(V, step)
	return k, V, Ex, Ey, err
#
def poly_5(h, z):
	# Returns the polynomial fit evaluated at h
	return z[0]*(h**5) + z[1]*(h**4) + z[2]*(h**3) + z[3]*(h**2) + z[4]*(h) + z[5]
#
# Argument Parsing Block
parser = arg.ArgumentParser()
parser.add_argument("--varyH", type=int, default=0, help="evaluate Q for h/Ly = {0.1, 0.2, ..., 0.9}")
params = parser.parse_args()
varyH = getattr(params, "varyH")
#
k, V, Ex, Ey, err = eval_V_E(h)		# Computing V, E and error
N_iter = len(err)	
start_fit = 3000	# log(err) behaves linearly after ~ 3000 iterations
params, __ = opt.curve_fit(fit_err, range(start_fit, N_iter), np.log(err[start_fit:]))
true_err = -(np.exp(params[1])/params[0])*np.exp(params[0]*(N_iter+0.5))	# Upper Bound on Cumulative Error
D2_D1 = Ey[k-1,:]/(eps_r*Ey[k+1,:] + 1e-15)		# Normal component of D
												# 1e-15 added in the denominator to avoid division by 0
theta0 = np.arctan2(Ey[k-1, :], Ex[k-1, :])		# Snell's Law falsification/verification
theta1 = np.arctan2(Ey[k+1, :], Ex[k+1, :]) 	# angle(E) just inside dielectric medium	
delta_theta = theta0 - theta1	
ratio = np.cos(theta0)/(np.cos(theta1) + 1e-15)
#
print("Maximum bound for error = {}".format(roundf(true_err, precision= 3)))
print("Last iteration value of error = {}".format(roundf(err[-1], precision= 3)))
print("Number of iterations computed = {}".format(N_iter))
print("Ratio of Normal Components of D just above and beneath the fluid interface: \
	\n Min. = {}\n Max. = {}\n Mean = {}".format(roundf(np.min(D2_D1), precision= 3),\
	 roundf(np.max(D2_D1), precision= 3), roundf(np.mean(D2_D1), precision= 3)))
#
if (varyH == 0):
	Q = np.sum(Ey[0, :]*step*eps0)	# Q_top for h = 0.5
	print("Capacitance at h = {}cm: {}F".format(h, roundf(Q, precision= 3)))
else:
	Q_top = []
	Q_fluid = []
	H = Ly*np.linspace(0.1, 0.9, 9)
	#H = Ly*np.linspace(0.1, 0.9, 27) 	# Taking more points gives better accuracy at the cost of speed
	
	for ht in H:
		k, __, Ex, Ey, __ = eval_V_E(ht, Nmax = 5000)
		Q_top.append(np.sum(Ey[0, :]*step*eps0))
		Q_fluid.append(-1*eps0*eps_r*step*(np.sum((Ex[k+1:, -1] - Ex[k+1:, 0]))+np.sum(Ey[-1, :])))
	Cap = Q_top.copy()		# C = Q/V, V = 1 volt
	# empirical C-h fit:
	fit_data = 1/np.sqrt(Cap)
	fit_coeffs = np.polyfit(H, fit_data, 5)	# Fitting 5th order polynomial to 1/sqrt(Q) (linear function of w_o) v/s h
	fit_C = poly_5(H, fit_coeffs)
	'''
		ALGORITHM TO FIND h GIVEN w_0, L AND L_z (depth along z-axis):

		Assumption: The system is invariant along z-axis, else, fringing will take place
					at the edges of the walls // to x-y plane
		w_0 = 1/sqrt(L*L_z*Cap) = poly_5(h, fit_coeffs)/sqrt(L*L_z)

		Let w_0 = poly_5(h, w_coeffs); w_coeffs are coefficients of polynomial map from h to w_0
		=> w_coeffs = fit_coeffs/sqrt(L*L_z)

		h = real root of poly_5(H, w_roots) - w_0 that belongs to the range (0, Ly)
		Root may be found using np.roots method
		 
		Note: Due to small ripple in the polynomial fit of 1/sqrt(C) versus h in the range (0, 0.6Ly), multiple roots
			  may be obtained in the significant range using this method. The actual mapping from w_0 to h must have 
			  atmost one root in (0, Ly) as the behaviour is monotonic. 
	'''
	print("Capacitance at h = {}cm: {}F".format(H[4], roundf(Cap[4], precision= 3)))
	#
	fig, axs = plt.subplots(2)
	axs[0].plot(H, Q_top)
	axs[0].set_title("Charge on +1V plate v/s h")
	axs[1].plot(H, Q_fluid)
	axs[1].set_title("Charge at fluid-wall interfaces v/s h")
	fig.tight_layout()
	plt.figure("C-h fit")
	plt.plot(H, 1/np.sqrt(Cap), "rx")
	plt.plot(H, 1/np.sqrt(Cap))
	plt.plot(H, fit_C)
	plt.legend(["data points", "point-wise linear interpolation", "polynomial fit of order 5"])
	plt.title(r"Polynomial fit of $\frac{1}{\sqrt{C}}$")
	plt.ylabel(r"$\frac{1}{\sqrt{C}}$"); plt.xlabel("h")
#
# Contour and Quiver Plots of V, E: 
y = np.linspace(0, Ly, M); x = np.linspace(0, Lx, N)
Y, X = np.meshgrid(y, x)
plt.figure("E_andV")
plt.quiver(x[:-1]+(step/2), y[:-1]+(step/2), Ex, Ey)
plt.axhline(Ly - h, color = 'r' )
CS = plt.contour(X, Y, V.T)
plt.title("Electric Field Profile and Equipotential Lines")
plt.ylabel("+1V plate <-- Y --> GND plate"); plt.xlabel("GND <-- X --> GND")
plt.figure("contourV")
plt.axhline(Ly - h, color = 'red')
plt.contourf(X, Y, V.T); plt.colorbar()
plt.title("Potential Profile inside the tank")
plt.ylabel("+1V plate <-- Y --> GND plate"); plt.xlabel("GND <-- X --> GND")
#
plt.figure("err_fit")
plt.plot(range(N_iter), np.log(err))
plt.plot(range(start_fit, N_iter), fit_err(range(start_fit, N_iter), params[0], params[1]))
plt.title("Comparison of log(error) with linear fit")
plt.ylabel("log(err)"); plt.xlabel("No. of iterations")
#
fig1, ax1 = plt.subplots(2)
ax1[0].plot(ratio)
ax1[0].set_title(r"Plot of $\frac{sin(\theta_0)}{sin(\theta_1)}$ versus x at fluid interface")
ax1[1].plot(delta_theta)
ax1[1].set_title(r"Change in angle of $\vec{E}$ at the fluid-air interface")
fig1.tight_layout()
plt.show()