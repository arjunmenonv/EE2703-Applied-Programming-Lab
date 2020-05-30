'''
		CONVOLUTION OF SIGNALS USING THE FAST FOURIER TRANSFORM ALGORITHM
		Author: Arjun Menon Vadakkeveedu
		Roll Number: EE18B104
		EE2703, Applied Programming Lab, Electrical Engineering, IIT MADRAS
		7th May 2020
		
		FILE INPUT FORMAT:
		usage: EE18B104_Assign10.py [-h] [--fn_choice FN_CHOICE]

		optional arguments:
		  -h, --help            show this help message and exit
		  --fn_choice FN_CHOICE
                        Choose function:
                        1: FIR Filter Characteristics	
                        2: Linear and Circular Convolution with FIR filter
                        3: Autocorrelation of Zadoff-Chu sequence
'''
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sp 
from numpy.fft import fft, fftshift, ifft, ifftshift
import pandas as pd 	# To read complex numbers from a csv file
import argparse as ag 
#
pi = np.pi
#
def csv_in(fname, i2j = 0):
	data = pd.read_csv(fname, sep=',', header= None)
	if i2j:
		data = data.applymap(lambda s: np.complex(s.replace('i', 'j'))).values
	x = np.array(data, dtype = complex).flatten()
	return x
def filter_char(b, a, show= True):
	# Pole-zero characteristics
	zero, pole, __ = sp.tf2zpk(b, a)
	if (pole.shape < zero.shape):
		pole = np.concatenate((pole, np.zeros(zero.shape[0] - pole.shape[0], dtype = complex)))
	zp = [zero, pole]
	# Magnitude and Phase Characteristics
	w1, H = sp.freqz(b, a, 1024, whole= True)
	# Time domain sequence (= Coefficients for given filter since it is FIR)
	h = ifft(H)
	ii = np.where(abs(h) > 1e-4)
	h = np.real(h[ii])	# remove noisy imaginary components- h is a real FIR filter
	# Group Delay of Filter
	w2, gd = sp.group_delay((b, a), whole= True)
	if show== True:
		plt.figure("Pole-Zero Plot")
		plt.polar(np.angle(zp[0]), np.abs(zp[0]), 'bo')
		plt.polar(np.angle(zp[1]), np.abs(zp[1]), 'gx')
		plt.polar(np.linspace(0, 2*pi, 360), np.ones(360), 'k-')
		plt.title("Pole-Zero Plot")
		plt.figure("Mag-Phase Characteristics")
		plt.subplot(211); plt.plot(w1, abs(H))
		plt.title("Mag-Phase Characteristics of FIR filter")
		plt.subplot(212); plt.plot(w1, np.unwrap(np.angle(H)))
		plt.figure("h[n]")
		plt.plot(h, "ro")
		plt.title("Time Domain: FIR Filter")
		plt.figure("Group Delay of Filter")
		plt.plot(w2, gd)
		plt.title("Group Delay of Filter")
	return zp, H, h, gd
#Block Convolution: Overlap and Save
def overlap_save(x, h, L=32, P=16):
	x_d = np.concatenate((np.zeros(P), x, np.zeros(L)))
	if h.shape != P:
		h_2m = np.concatenate((h, np.zeros(P - len(h))))
	n_os = (len(x)+L)/(L-P)
	y = np.zeros(len(x)+len(h_2m)+L, dtype = complex)
	for i in range(int(n_os)):
		y_d = ifft(fft(x_d[i*(L-P):i*(L-P) + L])*fft(np.concatenate((h_2m, np.zeros(L-P)))))
		y[i*(L-P):(i+1)*(L-P)] = y_d[P:L]
	y = y[0:len(x)+len(h)-1]
	return y
# Periodic Correlation in FFT Domain:
def fft_corr(x, y, circ_shift= 0):
	y = np.roll(y, circ_shift)
	Y = fft(y)
	X = fft(x)
	Y[1:] = np.flip(Y[1:])
	X[1:] = np.flip(X[1:])
	z = ifft(X*np.conj(Y))
	Z = fftshift(fft(z))
	return z, Z
def gen_subplot(x, y, xlm, title):
	fig = plt.figure(title[0])
	ax0 = plt.subplot(211); ax0.plot(x[0], y[0]); plt.xlim(xlm)
	ax0.set_title(title[1])
	ax1 = plt.subplot(212); ax1.plot(x[1], y[1]); plt.xlim(xlm)
	ax1.set_title(title[2])
	fig.tight_layout()
	plt.show()
	return 0
# Argument Parsing Block
parser = ag.ArgumentParser(formatter_class = ag.RawTextHelpFormatter)
parser.add_argument("--fn_choice", type=int, default=2, help="Choose function:\n1: FIR Filter Characteristics\
	\n2: Linear and Circular Convolution with FIR filter\n3: Autocorrelation of Zadoff-Chu sequence")
params = parser.parse_args()
fn_choice = getattr(params, "fn_choice")
H_coeff = csv_in('./csv files/h.csv')
if fn_choice == 1:
	zp, H, h, gd = filter_char(H_coeff, 1)
	plt.show()
elif fn_choice == 2:
	zp, H, h, gd = filter_char(H_coeff, 1, show= False)
	n = np.linspace(1, 1024, 1024)
	x = np.cos(0.2*pi*n) + np.cos(0.85*pi*n)
	# Linear Convolution:
	y_lin = np.convolve(x, h)
	n_ylin = np.linspace(1, len(y_lin), len(y_lin))
	Y_lin = fftshift(fft(y_lin))
	w_ylin = np.linspace(-pi, pi, len(Y_lin)+1)[:-1]
	# Circular Convolution:
	y_cir = np.real(ifft(fft(x)*H))
	#Block Convolution:
	y_block = np.real(overlap_save(x, h)) #L= 32, P= 16
	print("\n\nSum of Absolute Errors (lin and block) = ", np.sum(abs(y_block - y_lin)))
	# Plot:
	plt.figure("Input Sequence")
	plt.plot(n, x); plt.xlim([1, 256]); plt.title("Input Sequence")
	#
	plt.figure("Comparison: Linear versus Circular")
	plt.plot(n, y_cir - y_lin[:1024], 'rx'); plt.xlim([0, 25])
	plt.title("Comparison: Linear versus Circular")
	plt.figure("Block Convolution")
	plt.plot(y_block); plt.xlim([0, 256])
	plt.title("Block Convolution")	
	#
	title_t = ["Linear and Circular Convolution", "Time Domain: Linear Convolution", "Time Domain: Circular Convolution"]
	title_f = ["Linear and Circular Convolution", "FFT: Linear Convolution", "FFT: Circular Convolution"]
	gen_subplot([n_ylin, n], [y_lin, y_cir], [0, 256], title_t)
	gen_subplot([w_ylin, np.linspace(-pi, pi, len(y_cir)+1)[:-1]], [abs(Y_lin), abs(fftshift(fft(y_cir)))], [-5, 5], title_f)
elif fn_choice == 3:
	# Zadoff-Chu sequence:
	x1 = csv_in('./csv files/x1.csv', i2j= 1)
	delay = 5
	z, Z = fft_corr(x1, x1, delay)
	plt.figure("Autocorrelation of Zadoff-Chu Sequence")
	plt.plot(abs(z)); plt.title("Autocorrelation of Zadoff-Chu Sequence")
	plt.xlim([np.max([delay-10, 0]),np.min([delay+10, len(x1)])])
	plt.figure("FFT_z")
	plt.plot(abs(Z)); plt.title("FFT of z")
	plt.show()
#