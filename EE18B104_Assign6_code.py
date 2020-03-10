'''
		LAPLACE TRANSFORM
		Author: Arjun Menon Vadakkeveedu
		Roll Number: EE18B104
		EE2703 Applied Programming Lab, Electrical Engineering, IIT Madras
		7 March 2020
'''
from pylab import *
import scipy.signal as sp 
import os
import cv2
'''
	For exponentially decaying sinusoidal input functions, ie functions of the form f(t) = cos(wt).exp(-at)u(t), the Laplace transform is:
	F(s) = (s+a)/((s+a)^2 + w^2) = Num/(Num^2 + freq^2)
	#
	A Second Order Linear Differential Equation systems with constant coefficients can be represented in s domain as:
	s^2.X(s) + 2.(zeta).w.sX(s) + w^2.X(s) = F(s)
	where zeta determines the damping (zeta<1 => underdamped; zeta=1 => critically damped; z>1 => overdamped)
	#
	X(s) = F(s)/(s^2 + 2.(zeta).w.s + w^2) = F(s)/H(s)
'''
def exp_sin_resp(decay, zeta, freq, tvec):
	num = poly1d([1, decay])
	den = polyadd(polymul(num, num), poly1d([freq*freq]))
	den_x = poly1d([1, 2*zeta*freq, freq*freq])
	den = polymul(den, den_x)
	X = sp.lti(num.coeffs, den.coeffs)	#num and den are poly1d objects, poly1d.coeffs is an array with the polymonial coefficient values 
	t, x = sp.impulse(X, None, tvec)
	plot(t, x)
	title("Decay factor = "+str(decay))
	savefig("./Images/expsin_" + str(decay) +".png")
	close()
	return 0
def exp_sin_respLTI(decay, zeta, freq, tvec):
	H = sp.lti([1], poly1d([1, 2*zeta*1.5, 2.25]).coeffs)
	u = cos(freq*tvec)*exp(-1*decay*tvec)
	t, x, svec = sp.lsim(H, u, tvec)
	plot(t, x)
	title("Complete Response at "+str(freq)+ "Hz")
	savefig("./Images/expsinLTI_" + str(freq) + ".png")
	close()
	return 0
def gen_subplots(t, p1, p2, title1, title2, plot_type, fname):
	fig, axs = subplots(2)
	axs[0].set_title(title1)
	axs[1].set_title(title2)
	if(plot_type == "semilogx"):
		axs[0].semilogx(t, p1)
		axs[1].semilogx(t, p2)
	elif(plot_type == "semilogy"):
		axs[0].semilogy(t, p1)
		axs[1].semilogy(t, p2)
	elif(plot_type == "loglog"):
		axs[0].loglog(t, p1)
		axs[1].loglog(t, p2)
	else:
		axs[0].plot(t, p1)
		axs[1].plot(t, p2)
	fig.tight_layout()
	savefig(fname)
	close()
	return 0
def LinCktResp(u, t, tname, fname):
	t, x, svec = sp.lsim(H, u, t)
	plot(t, x)
	title(tname)
	savefig(fname)
	close()
	return 0
#
exp_sin_resp(0.5, 0, 1.5, linspace(0, 50, 501))	
exp_sin_resp(0.05, 0, 1.5, linspace(0, 50, 501))	
# 
for freq in arange(1.4, 1.65, 0.05):	
	exp_sin_respLTI(0.05, 0, freq, linspace(0, 50, 501))
#
den_y = poly1d([1, 0, 3, 0, 0])
num_y = poly1d([2, 0])
X = sp.lti([1, 0, 2, 0], den_y)
Y =sp.lti(num_y, den_y)
t, x = sp.impulse(X, None, linspace(0, 20, 501))
t, y = sp.impulse(Y, None, linspace(0, 20, 501))
#
H = sp.lti([1], [1e-12, 1e-4, 1])	# H(s) for the LCR network is 1/(1 + sCR + s^2LC)
w, S, phi = H.bode()
gen_subplots(t, x, y, "x(t)", "y(t)", "plot", "./Images/coupled_eqn.png")
gen_subplots(w, S, phi, "Magnitude Bode Plot in dB", "Phase Bode Plot in degree", "semilogx", "./Images/LinCktBode.png")
#
t_tran = arange(0, 1e-5, 1e-8)
u = cos(1e3*t_tran) - cos(1e6*t_tran)
LinCktResp(u, t_tran, "Transient Response", "./Images/LinCktTranResp.png")
t_comp = arange(0, 1e-2, 1e-8)
u = cos(1e3*t_comp) - cos(1e6*t_comp)
LinCktResp(u, t_comp, "Complete Response", "./Images/LinCktCompResp.png")
print("Do you want to view the plots? [y]/[Any other key]:")
press_key = input()
if (press_key == "y"):
	files = os.listdir('./Images')
	print("Press any key on the plot windows to terminate the program, DO NOT MANUALLY CLOSE THE WINDOWS! (unresolved bug with openCV function)")
	for f in files:
		cv2.imshow(str(f.split('.p')[0]), cv2.imread("./Images/" + str(f)))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

