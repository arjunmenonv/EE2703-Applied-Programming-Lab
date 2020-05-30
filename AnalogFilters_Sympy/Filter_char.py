'''
		EE2703: Applied Programming Lab- Symbolic Python for Solving Circuits
		Author: Arjun Menon Vadakkeveedu
		Date: 15 March 2020
'''
import sympy as sym
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
#
G = 1.586
#

def lpf(R1, R2, C1, C2, G):
	A = sym.Matrix([[0, 0, 1, -1/G], [-1/(1 + s*R2*C2), 1, 0, 0], [0, -G, G, 1], [(-1/R1 -1/R2 - s*C1), 1/R2, 0, s*C1]])
	B = sym.Matrix([0, 0, 0, 1/R1])
	V = A.inv()*B
	Vo = V[3]
	Vo = sym.simplify(Vo)
	num_H, den_H = sym.fraction(Vo)
	n_p = sym.Poly(num_H, s).all_coeffs()
	d_p = sym.Poly(den_H, s).all_coeffs()
	num = np.array(n_p, dtype = float)
	den = np.array(d_p, dtype = float)
	H = sp.lti(num, den)
	return (Vo, H)
def hpf(R1, R3, C1, C2, G):
	A = sym.Matrix([[0, 0, G, -1], [0, -G, G, 1], [s*C2*R3, -1 - s*C2*R3, 0, 0], [(1/R1 + s*C1 + s*C2), -s*C2, 0, -1/R1]])
	B = sym.Matrix([0, 0, 0, s*C1*1])
	V = A.inv()*B
	Vo = V[3]
	Vo = sym.simplify(Vo)
	num_H, den_H = sym.fraction(Vo)
	n_p = sym.Poly(num_H, s).all_coeffs()
	d_p = sym.Poly(den_H, s).all_coeffs()
	num = np.array(n_p, dtype = float)
	den = np.array(d_p, dtype = float)
	H = sp.lti(num, den)
	return (Vo, H)
def gen_subplots(plt_type, plt_vec, plt_title):
	f, ax = plt.subplots(2)
	if (plt_type[0] == "semilogx"):
		ax[0].semilogx(plt_vec[0], plt_vec[1])
	else:
		ax[0].plot(plt_vec[0], plt_vec[1])
	if (plt_type[1] == "semilogx"):
		ax[1].semilogx(plt_vec[2], plt_vec[3])
	else:
		ax[1].plot(plt_vec[2], plt_vec[3])
	ax[0].set_title(plt_title[0])
	ax[1].set_title(plt_title[1])
	f.tight_layout()
	return f
# main:
s = sym.symbols('s')
Vo_l, H_lpf = lpf(10e3, 10e3, 10e-12, 10e-12, G)	# H is the frequency domain impulse response
w_lpf, S_lpf, phi_lpf = H_lpf.bode()
Vo_h, H_hpf = hpf(10e3, 10e3, 1e-9, 1e-9, G)
w_hpf, S_hpf, phi_hpf = H_hpf.bode()
f_bode = gen_subplots(["semilogx", "semilogx"], [w_lpf, S_lpf, w_hpf, S_hpf], ["LPF Magnitude Response", "HPF Magnitude Response"])
#
print(Vo_l)
print(Vo_h)
#
t_l1 = np.arange(0, 2, 2e-3)
t_l2 = np.arange(0, 2e-3, 2e-6)
u_l1 = np.ones(t_l1.shape)
# lsim determines the convolution of h = ilaplace(H) with causal signals u, here u is a heaviside fn and o/p is calculated for the first 20 units of t
t_l1, y_l1, __ = sp.lsim(H_lpf, u_l1, t_l1)
u_l2 = np.sin(2000*np.pi*t_l2) + np.cos(2e6*np.pi*t_l2)
t_l2, y_l2, __ = sp.lsim(H_lpf, u_l2, t_l2)
f_lpf = gen_subplots(["plot", "#plot"], [t_l1, y_l1, t_l2, y_l2], ["LPF: Step Response", "LPF: Response to Sinusoid"])
#
t_h1 = np.arange(0, 2e-5, 2e-8)
t_h2 = np.arange(0, 2e-1, 2e-4)
u_h2 = np.ones(t_h2.shape)
u_h1 = np.cos(2e6*np.pi*t_h1)*np.exp(-1*t_h1)
t_h1, y_h1, __ = sp.lsim(H_hpf, u_h1, t_h1)
t_h2, y_h2, __ = sp.lsim(H_hpf, u_h2, t_h2)
f_hpf = gen_subplots(["plot", ""], [t_h2, y_h2, t_h1, y_h1], ["HPF: Step Response", "HPF: Response to Damped Sinusoid"])
# Low Frequency Inputs
t_misc = np.arange(0, 2e-1, 2e-4)
u_misc = np.cos(40*np.pi*t_misc) + np.sin(np.pi*t_misc)*np.exp(-1e-2*t_misc)
t_misc, y_lmisc, __ = sp.lsim(H_lpf, u_misc, t_misc)
t_misc, y_hmisc, __ = sp.lsim(H_hpf, u_misc, t_misc)
f_misc = gen_subplots(["plot", ""], [t_misc, y_lmisc, t_misc, y_hmisc], ["LPF: Response to Damped Sinusoid", "HPF: Response to Damped Sinusoid"])
plt.show()
