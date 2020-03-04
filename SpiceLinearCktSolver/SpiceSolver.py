'''
		Applied Programming Lab Assignment 2
		Linear Circuit Solver Based on SPICE

		Description: The following code accepts a SPICE netlist description of a linear circuit involving voltage sources, current sources,
		resistors, inductors and capacitors and solves them for both DC and AC excitations (of a single frequency)

		Author: Arjun Menon Vadakkeveedu
		Roll No.: EE18B104, Dual Degree in Electrical Engineering, IIT Madras
		Date: February 2, 2020
		
		Points to Note:
		1. Convention used for Current Flow- Let a DC current source be defined as:
		I_eg node_p node_m value
		then it is assumed that a positive current flows from node_p to node_m. 
		The same convention applies to AC current sources
		2. To represent complex numbers, numpy.complex128 (shorthand- np.complex) is used. These are double floating point precision complex
		numbers. Mathematical functions with complex inputs were implemented using the python library 'cmath'
		3. The node GND is assumed to have zero potential. If the netlist contains no node with the name 'GND' then the first node defined 
		is set to zero potential.
'''
import numpy as np 
from sys import argv, exit
import cmath
#
pi = 3.14159265359
# Class Definitions:
class passive:
	'''
	Class for linear passive components (R, L, C)
	For resistors, impedance = value
		inductors, impedance= j*(2*pi*f)*value
		capacitors, impedance = -j/(2*pi*f*value)
	The admittance (=1/impedance) value is passed while creating object from the function gen_obj
	'''
	def __init__(self, name, node_p, node_m, Y):
		self.name = name
		self.node_p = node_p
		self.node_m = node_m
		self.Y = Y
#
class V_src:
	is_ac = 0
	frequency = 0
	mag = 0
	phase = 0
	dc_val = 0
	'''
	While creating a V_src object, the function gen_obj passes a list 'properties' which is defined as:
		properties = [is_ac = 0, dc_val]	; if the V_src is a dc source
		properties = [is_ac = 1, frequency, mag, phase]
	Currently, this class only models AC and DC voltage sources; AC voltage sources with a DC offset are not considered here, 
	as they are equivalent to a superposition of a DC source and an AC source connected across the same nodes in series
	'''
	def __init__(self, name, node_p, node_m, properties = []):
		self.name = name
		self.node_p = node_p
		self.node_m = node_m
		self.is_ac = properties[0]
		if (self.is_ac):
			self.frequency = properties[1]
			self.mag = properties[2]
			self.phase = properties[3]
		else:
			self.dc_val = properties[1]
#
class I_src:
	is_ac = 0
	frequency = 0
	mag = 0
	phase = 0
	dc_val = 0
	'''
	Properties of this class are similar to that of V_src
	'''
	def __init__(self, name, node_p, node_m, properties = []):
		self.name = name
		self.node_p = node_p
		self.node_m = node_m
		self.is_ac = properties[0]
		if (self.is_ac):
			self.frequency = properties[1]
			self.mag = properties[2]
			self.phase = properties[3]
		else:
			self.dc_val = properties[1]
#
#	FUNCTION DEFINITIONS:
def file_extract():
	# File I/O Function. Reads the netlist and removes irrelevant information. Returns lists lines (circuit definition) and ac (ac characteristics, if any)
	if len(argv) != 2:
		print("Invalid Input Format; Expected Input as 'python3 ee18b104_spice2.py <netlist_name>'")
		exit()
	try:
		with open(argv[1]) as file_obj:
			lines = file_obj.readlines()
	except FileNotFoundError:
		print("File Not Found; Exiting")
		exit()
	str_beg = ".circuit"; str_end = ".end"; str_comm = "#"; car_ret = "\n"; ac_str = ".ac ";
	beg_idx = -1; end_idx = -2 	
	# raises error if str_beg and str_end are not found
	lines = [lines[i].split(car_ret)[0].split(str_comm)[0].strip() for i in range(len(lines))] 
	# Removes unnecessary characters
	lines = list(filter(None, lines))
	# Removes empty elements in lines
	ac = []
	ac.append([line.split() for line in lines if (line[0:len(ac_str)] == ac_str)]) 
	for i in range(len(lines)):
		if str_beg == lines[i][0:len(str_beg)]:
			beg_idx = i
			break
	for i in range(beg_idx+1, len(lines)):
		if str_end == lines[i][0:len(str_end)]:
			end_idx = i
			break
	if ((beg_idx>=end_idx) or (beg_idx == -1)):
		print("Invalid circuit definition, exiting")
		exit()
	lines = [lines[i].split() for i in range(beg_idx+1, end_idx)]
	return lines, ac
#
def gen_obj(comp_def, frequency):
	#Function to assign objects of a particular class 
	obj_type = comp_def[0][0]
	if (obj_type == "R"):
		dummy_obj = passive(comp_def[0], comp_def[1], comp_def[2], 1/float(comp_def[3])) 
		#LIST ELEMENTS ARE STRINGS, 'values' ARE CONVERTED TO FLOAT
		psv.append(dummy_obj)
	elif (obj_type == "L"):
		if (not frequency):
			dummy_obj = V_src(comp_def[0]+"_shrt", comp_def[1], comp_def[2], [0, 0])
			#Connect a 0 V source across node_p & node_m for L in DC
		else:
			dummy_obj = passive(comp_def[0], comp_def[1], comp_def[2], -1j/(2*pi*frequency*float(comp_def[3])))	#admittance 
			psv.append(dummy_obj)
	elif (obj_type == "C"):
		dummy_obj = passive(comp_def[0], comp_def[1], comp_def[2], 1j*(2*pi*frequency*float(comp_def[3]))) 
		#Capacitors in DC are automatically open circuited when f = 0 
		psv.append(dummy_obj)
	elif (obj_type == "V"):
		if comp_def[3] == "ac":
			is_ac = 1; mag = float(comp_def[4]);
			try:
				phase = float(comp_def[5])
			except:
				phase = 0
			properties = [is_ac, frequency, mag, phase]
		else:
			is_ac = 0; dc_val = float(comp_def[-1])
			properties = [is_ac, dc_val]
		dummy_obj = V_src(comp_def[0], comp_def[1], comp_def[2], properties)
		V_s.append(dummy_obj)
	elif (obj_type == "I"):
		if comp_def[3] == "ac":
			is_ac = 1; mag = float(comp_def[4]);
			try:
				phase = float(comp_def[5])
			except:
				phase = 0
			properties = [is_ac, frequency, mag, phase]
		else:
			is_ac = 0; dc_val = float(comp_def[-1])
			properties = [is_ac, dc_val]
		dummy_obj = I_src(comp_def[0], comp_def[1], comp_def[2], properties)
		I_s.append(dummy_obj)
	else:
		print("Invalid Component Definition:", comp_def, "\nExiting...")
		exit()
	return 0
#
def node_search(arr):
	# Generates a list nodes that contains all node names
	for i in range(len(arr)):
		for comp in arr[i]:
			node_name = comp.node_p
			flag = 0
			for node in nodes:
				if(node_name == node):
					flag = -1
					break
			if (flag == 0):
				nodes.append(node_name)
			node_name = comp.node_m
			flag = 0
			for node in nodes:
				if(node_name == node):
					flag = -1
					break
			if (flag == 0):
				nodes.append(node_name)
	for node in nodes:
	 # If a node named "GND" exists, make GND the first node- to add the condition that V(GND) = 0
		if(node == "GND"):
			idx = nodes.index(node)
			nodes[0], nodes[idx] = node, nodes[0]
	return nodes
#
def gen_B(nodes, I_s, V_s):
	# Generates a column vector B containing the known terms- Independent currents at nodes and potentials of independent V sources 
	B = np.zeros(n+k, dtype = complex)
	for i in range(1, len(nodes)):
		i_p = 0; i_m = 0
		for src in I_s:
			if(nodes[i] == src.node_p):
				current = src.dc_val + src.mag*cmath.exp(1j*src.phase*pi/180)
				# Since these ckts don't have superposition of DC and AC, at most one of the two terms will be non-zero at all times
				i_p -= current
			elif(nodes[i] == src.node_m):
				current = src.dc_val + src.mag*cmath.exp(1j*src.phase*pi/180)
				i_m += current
		B[i] = i_p + i_m
	for i in range(len(V_s)):
		B[n + i] = V_s[i].dc_val + V_s[i].mag*cmath.exp(1j*V_s[i].phase*pi/180)
	return B
#
def gen_A(nodes, psv, V_s):
	# Generates coefficient matrix
	A = np.zeros((n+k, n+k), dtype = complex)
	for i in range(len(nodes)):
		for comp in psv:
			if(nodes[i] == comp.node_p):
				node_2 = comp.node_m
				A[i][i] += comp.Y
				A[i][nodes.index(node_2)] -= comp.Y
			elif(nodes[i] == comp.node_m):
				node_2 = comp.node_p
				A[i][i] += comp.Y
				A[i][nodes.index(node_2)] -= comp.Y
		for comp in V_s:
			if(nodes[i] == comp.node_p):
				V_pos = V_s.index(comp)
				A[n+V_pos][i] += 1
				A[i][n+V_pos] += 1
			elif(nodes[i] == comp.node_m):
				V_pos = V_s.index(comp)
				A[n+V_pos][i] -= 1
				A[i][n+V_pos] -= 1
	A[0][0] = 1	
	# Replacing first equation (which is redundant) with V_GND = 0
	A[0][1:n+k] = 0
	return A
#
try:
	circ_def, ac_char  = file_extract()
	try:
 		frequency = float(ac_char[0][0][2])
	except IndexError:
		frequency = 0
	# NOTE: Since the circuits modelled by this program are excited only for one frequency in ac mode, only the first element of the ac_char list
	#		is relevant- the information for AC voltage sources is anyways provided by the circuit definition. Thus, only the frequency info is
	#		extracted from ac_char. The complete list ac_char is still extracted from the file since it is required for circuits with multiple 
	#		frequency excitations, which is a feature that may be added on to this.
	psv = []; V_s = []; I_s = []
	nodes = []	# Each element of nodes contains node name
	[gen_obj(comp_def, frequency) for comp_def in circ_def]	
	nodes = node_search([psv, V_s, I_s])
	n = len(nodes); k = len(V_s)
	B = gen_B(nodes, I_s, V_s)
	A = gen_A(nodes, psv, V_s)
	x_val = np.linalg.solve(A, B)
	x_var = nodes + [V.name for V in V_s]
	print("Variable matrix = \n", x_var, "\nSolution = ")
	for x in x_val:
		print(x, "\n")
except Exception as z:
	print("Exception! Please rectify the issue:\n", z)
