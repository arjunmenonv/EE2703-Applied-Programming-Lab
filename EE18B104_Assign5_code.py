'''
		THE RESISTOR PROBLEM: SOLVING LAPLACE'S EQUATION ITERATIVELY (FINITE DIFFERENCE METHOD)IN A CONDUCTOR GIVEN A POTENTIAL PROFILE
		Author: Arjun Menon Vadakkeveedu
		Roll Number: EE18B104
		EE2703 Applied Programming Lab, Electrical Engineering, IIT Madras
		3 March 2020
'''
from pylab import *
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from sys import argv, exit
import argparse
from scipy.optimize import curve_fit
# Argument Parsing Block
parser = argparse.ArgumentParser()
parser.add_argument("--Nx", type=int, default=25, help="size along x (default = 25)")
parser.add_argument("--Ny", type=int, default=25, help="size along y (default = 25)")
parser.add_argument("--radius", type=float, default=0.35, help="radius of central lead (< 1) (default = 0.35)")
parser.add_argument("--Niter", type=int, default=1500, help="number of iterations (default = 1500)")
params = parser.parse_args()
Nx = getattr(params, "Nx"); Ny = getattr(params, "Ny"); radius = getattr(params, "radius"); Niter = getattr(params, "Niter")
#
def fit_err(k, slope, intercept):
	return k*slope + intercept
#
x = linspace(-0.5, 0.5, Nx)	# in scale 0.1mm (considering sheet to be 10mmx10mm)
y = linspace(-0.5, 0.5, Ny)
phi = zeros((Ny, Nx))
Jx = zeros((Ny, Nx))
Jy = zeros((Ny, Nx))
Y, X = meshgrid(y, x)
ii = where(X*X + Y*Y <= radius*radius)
phi[ii] = 1.00
xn, yn = ii
plt.plot(x[xn], y[yn], "ro")
plt.contourf(X, Y, phi)
plt.title("Contour plot of phi before solving")
errors = zeros(Niter)
for k in range(Niter):
	oldphi = phi.copy()
	phi[1:-1, 1:-1] = 0.25*(phi[1:-1, 0:-2] + phi[1:-1, 2:] + phi[0:-2, 1:-1] + phi[2:, 1:-1])	#Poisson Update
	phi[1:-1, 0] = phi[1:-1, 1]	# Boundary Condition for Left Surface
	phi[1:-1, -1] = phi[1:-1, -2]	# Boundary Condition for Right Surface
	phi[0, 1:-1] = phi[1, 1:-1]		# Boundary Condition for Top Surface
	phi[ii] = 1.0
	errors[k] = (abs(phi - oldphi)).max()
#
popt1, __ = curve_fit(fit_err, range(Niter), log(errors))
popt2, __ = curve_fit(fit_err, range(500, Niter), log(errors[500:]))
#
true_err = (popt2[1]/popt2[0])*exp(popt2[0]*(Niter+0.5))	# error in log scale
print("Maximum bound for error = ", true_err)
print("Last iteration value of error = ", errors[-1])
# Error in semilogy and log-log scales
fig_err, axs = plt.subplots(2)
err = axs[0].semilogy(range(Niter)[::50], errors[::50],'ro')
axs[0].set_title("Semilog scale error in consecutive iterations of phi")
axs[1].loglog(range(Niter)[::50], errors[::50],'ro')
axs[1].set_title("Log-log scale error in consecutive iterations of phi")
fig_err.tight_layout()
plt.show()
# Errors and error-fit
fig_fit = plt.plot(range(Niter), log(errors))
plt.plot(range(Niter), fit_err(range(Niter), popt1[0], popt1[1]))
plt.plot(range(500, Niter), fit_err(range(500, Niter), popt2[0], popt2[1]))
plt.title("Error and Curve fits of error")
plt.legend(["err", "fit1", "fit2"])
plt.show()
# Contour + Quiver 
plt.plot(y[yn], x[xn], "ro")
plt.contour(X, Y, phi.T)
Jx[1:-1, 1:-1] = 0.5*(phi[1:-1, 0:-2] - phi[1:-1, 2:])
Jy[1:-1, 1:-1] = 0.5*(phi[0:-2, 1:-1] - phi[2:, 1:-1])
plt.quiver(x, y, Jx[::1, :], Jy[::1, :])
plt.show()
#	Surface Plot of Potential
fig1=figure()
ax=p3.Axes3D(fig1) # Axes3D is the means to do a surface plot
title('The 3-D surface plot of the potential')
surf = ax.plot_surface(Y, X, phi.T, rstride=1, cstride=1, cmap=cm.jet)
show()
#

