import numpy as np 
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftshift, ifftshift
import argparse as ag

pi = np.pi
# Lambda function definitions:
sin = lambda t: np.sin(5*t)
amp_mod = lambda t: (1+0.1*np.cos(t))*np.cos(10*t)
f_sin3 = lambda t: np.sin(t)**3
f_cos3 = lambda t: np.cos(t)**3
freq_mod = lambda t: np.cos(20*t + 5*np.cos(t))
gauss_t = lambda t: np.exp(-(t**2)/2)
gauss_f = lambda w: np.exp(-((w**2)/2))*(np.sqrt(2*pi))	
#
def gen_fft(f, num_cycles, sample_freq, ii_cutoff):	
	N = num_cycles*sample_freq
	x = np.linspace(-1*pi, pi, N+1)*num_cycles; x = x[:-1]
	y = f(x)
	y = ifftshift(y)
	Y = fftshift(fft(y))/N
	w = np.linspace(-0.5, 0.5, N+1)*sample_freq; w = w[:-1]
	ii = np.where(abs(Y)>ii_cutoff)
	return Y, w, ii
def gen_plots(Y, w, ii, x_range, title, plot_gauss = False):
	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.xlim(x_range)
	plt.plot(w, abs(Y), lw = 2)
	if plot_gauss == True:
		plt.plot(w, gauss_f(w))
		plt.legend(["Estimated FFT", "True FFT"])
	plt.ylabel(r"|Y|", size = 16)
	plt.grid(True)
	plt.title(title)
	plt.subplot(2,1,2)
	plt.xlim(x_range)
	plt.plot(w[ii], (np.angle(Y[ii])), 'ro', lw = 2)
	plt.ylabel(r"Phase of Y")
	plt.grid(True)
	plt.tight_layout()
	plt.show()
	if plot_gauss == True:
		plt.xlim(x_range)
		plt.plot(w, np.real(Y - gauss_f(w)))
		plt.title("Error of DFT with true Fourier Transform")
		plt.show()
	return 0
# Argument Parsing Block
parser = ag.ArgumentParser(formatter_class = ag.RawTextHelpFormatter)
parser.add_argument("--fn_choice", type=int, default=5, help="Choose time domain function:\n1: sin(5t)\
	\n2: (1+0.1cos(t))cos(10t) (AMPLITUDE MODULATION)\n3: sin^3(t), cos^3(t)\n4: cos(20t + 5cos(t)) (FREQUENCY MODULATION)\
	\n5: exp(-(t^2)/2) (GAUSSIAN)")
params = parser.parse_args()
fn_choice = getattr(params, "fn_choice")
sampling_freq = 64
num_cycles = 4
if fn_choice == 1:
	Y, w, ii = gen_fft(sin, num_cycles, sampling_freq, 1e-3)
	gen_plots(Y, w, ii, [-15, 15], "Spectrum of sin(5t)")
elif fn_choice == 2:
	Y, w, ii = gen_fft(amp_mod, num_cycles, sampling_freq, 1e-3)
	gen_plots(Y, w, ii, [-15, 15], "Spectrum of (1+0.1cos(t))cos(10t) (AM signal)")
elif fn_choice == 3:
	Y1, w1, ii1 = gen_fft(f_sin3, num_cycles, sampling_freq, 1e-3)
	gen_plots(Y1, w1, ii1, [-15, 15], "Spectrum of sin^3(t)")
	Y2, w2, ii2 = gen_fft(f_cos3, num_cycles, sampling_freq, 1e-3)
	gen_plots(Y2, w2, ii2, [-15, 15], "Spectrum of cos^3(t)")
elif fn_choice == 4:
	Y, w, ii = gen_fft(freq_mod, num_cycles, sampling_freq, 1e-3)
	gen_plots(Y, w, ii, [-40, 40], "Spectrum of cos(20t + 5cos(t)) (FM signal)")
elif fn_choice == 5:
	sampling_freq = 128
	num_cycles = 16
	Y, w, ii = gen_fft(gauss_t, num_cycles, sampling_freq, 1e-6)
	Y = Y*num_cycles*2*pi
	gen_plots(Y, w, ii, [-6*pi, 6*pi], "Spectrum of Gaussian", True)
#