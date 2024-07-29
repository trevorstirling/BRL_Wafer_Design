#########################################################################
# BRL Design Functions                                                  #
# - Refractive_Index                                                    #
# - colours                                                             #
# - Effective_Nonlinearity                                              #
# - Matching_Layer_Thickness                                            #
# - Find_Mode_Near                                                      #
# - Mode_Field                                                          #
# - Far_Field_Profile                                                   #
#                                                                       #
# Author: Trevor Stirling                                               #
# Date: July 6, 2024                                                    #
#########################################################################

import os
from datetime import datetime
import matplotlib.pyplot as plt
from cmath import exp, sqrt, sinh, tanh, log, cos, sin, pi, atan
import numpy as np

def Refractive_Index(wl, material, model="Default"):
	#########################################################################
	# Finds the refractive index of a material at a given wavelength        #
	#                                                                       #
	# Inputs:                                                               #
	# wl is the wavelength [m]                                              #
	# material is a string containing the name of the material of interest  #
	#                                                                       #
	# Outputs:                                                              #
	# n_r is the real part of the refractive index                          #
	# n_i is the imaginary part of the refractive index                     #
	#########################################################################
	if material == "Air":
		nr = 1
		ni = 0
	elif material == "AlN": #From Waterloo - Ellipsometry by Trevor - "RF2", 16sccm N2, 25.6 sccm Ar, 103mm
		nr = 1.997+0.00464/(wl*1e6)**2+0.00140/(wl*1e6)**4
		ni = 0
	elif material == "TiO_2": #From Waterloo - Ellipsometry by Trevor - 27sccm Ar, 120mm
		nr = 1.232+0.31635/(wl*1e6)**2-0.01150/(wl*1e6)**4
		ni = 0
	elif material == "HfO_2": #From Devore 1951 - crystalline
		nr = sqrt(1.9558*(wl*1e6)**2/((wl*1e6)**2-0.15494**2)+1.345*(wl*1e6)**2/((wl*1e6)**2-.0634**2)+10.41*(wl*1e6)**2/((wl*1e6)**2-27.12**2)+1)
		ni = 0
	elif material == "Al_2O_3": #From Malitson and Dodge 1972
		nr = sqrt(1.4313493*(wl*1e6)**2/((wl*1e6)**2-0.0726631**2)+0.65054713*(wl*1e6)**2/((wl*1e6)**2-0.1193242**2)+5.3414021*(wl*1e6)**2/((wl*1e6)**2-18.028251**2)+1)
		ni = 0
	elif material == "SiO_2": #From Malitson 1965
		nr = sqrt(1+0.6961663*(wl*1e6)**2/((wl*1e6)**2-0.0684043**2)+0.4079426*(wl*1e6)**2/((wl*1e6)**2-0.1162414**2)+0.8974794*(wl*1e6)**2/((wl*1e6)**2-9.896161**2))
		ni = 0
	elif material == "SiC": #From Singh et al. 1971 - alpha SiC
		nr = sqrt(1+5.5394*(wl*1e6)**2/((wl*1e6)**2-.026945))
		ni = 0
	elif material == "Si_3N_4": #From Luke et al. 2015
		nr = sqrt(1+3.0249*(wl*1e6)**2/((wl*1e6)**2-0.1353406**2)+40314*(wl*1e6)**2/((wl*1e6)**2-1239.842**2))
		ni = 0
	elif material == "SiO":
		nr = 1.8254+0.06398*exp(-(wl*1e9)/1295.86625)+0.18695*exp(-(wl*1e9)/270.39606)+0.89665*exp(-(wl*1e9)/270.46115)
		ni = 5.92935e-5+6.93776*exp(-(wl*1e9)/80.49947)+6.93776*exp(-(wl*1e9)/80.49947)+6.93776*exp(-(wl*1e9)/80.49947)
	elif material == "Si": #From Salzberg and Villa 1957
		nr = sqrt(1+10.6684293*(wl*1e6)**2/((wl*1e6)**2-0.301516485**2)+0.003043475*(wl*1e6)**2/((wl*1e6)**2-1.13475115**2)+1.54133408*(wl*1e6)**2/((wl*1e6)**2-1104.0**2))
		ni = 0
	elif material == "MgO": #From Stephens and Malitson 1952
		nr = sqrt(2.956362+0.02195770/((wl*1e6)**2-0.01428322)-0.01062387*(wl*1e6)**2-0.0000204968*(wl*1e6)**4)
		ni = 0
	elif material == "ZnO": #From Waterloo - Ellipsometry by Trevor - 120mm
		nr = 1.904+0.01857/(wl*1e6)**2+0.00442/(wl*1e6)**4
		ni = 0
	elif material[:10] == "AlGaInAsQW":
		x = float(material.split("{")[1].split("}")[0])
		if model == "Gehrsitz":
			nr, ni = Refractive_Index(wl, "AlGaAs_{"+str(x)+"}", "Gehrsitz")
		elif model == "Djurisic" or model == "Default": #From Djurisic 1999 - Note moodel is for AlGaAs but we use it for AlGaInAs
			if wl > 900e-9:
				#For long wavelengths, use Gehrsitz instead
				nr, ni = Refractive_Index(wl, "AlGaAs_{"+str(x)+"}", "Gehrsitz")
			else:
				E = 6.62607015e-34*299792458/wl/1.602176634e-19
				#input parameters from the reference
				Table_I = [[1.410,1.583,0.2242,-1.4235],[1.746,1.455,0.1931,-1.2160],[2.926,0.962,-0.2124,-0.7850],[3.170,0.917,-0.0734,-0.9393]]
				Table_II=[[1.347,0.02,-0.568,4.210],[3.06,14.210,-0.398,4.763],[0.0001,0.0107,-0.0187,0.3057],[3.960,1.617,3.974,-5.413],[6.099,4.381,-4.718,-2.510],[0.001,0.103,4.447,0.208],[1.185,0.639,0.436,0.426],[0.473,0.770,-1.971,3.384],[0.194,0.125,-2.426,8.601],[0.018,0.012,0.0035,0.310],[4.318,0.326,4.201,6.719],[0.496,0.597,-0.282,-0.139],[0.014,0.281,-0.275,-0.569],[4.529,4.660,0.302,0.241],[4.924,5.483,-0.005,-0.337],[0.800,0.434,0.572,-0.553],[0.032,0.052,-0.300,0.411],[4.746,4.710,-0.007,-0.565],[3.529,4.672,-6.226,0.643],[0.302,0.414,-0.414,1.136],[0.004,0.023,-0.080,0.435],[4.860,4.976,-0.229,0.081]]
				eq_12_vec = [1,x,x*(1-x),x**2*(1-x)]
				cubic_x_vec = [1-x,x,x*(1-x),x**2*(1-x)]
				E_vec = [sum([Table_I[i][j]*eq_12_vec[j] for j in range(len(eq_12_vec))]) for i in range(len(Table_I))]
				parameter_vec = [sum([Table_II[i][j]*cubic_x_vec[j] for j in range(len(cubic_x_vec))]) for i in range(len(Table_II))]
				#Equation 1
				f = lambda y: y**(-2)*(2-(1+y)**0.5-(1-y)**0.5)
				kai0=(E+1j*parameter_vec[2]*exp(-parameter_vec[3]*((E-E_vec[0])/parameter_vec[2])**2))/E_vec[0]
				kai0s=(E+1j*parameter_vec[2]*exp(-parameter_vec[3]*((E-E_vec[0])/parameter_vec[2])**2))/E_vec[1]
				epsilon_I=parameter_vec[1]*E_vec[0]**(-3/2)*(f(kai0)+0.5*(E_vec[0]/E_vec[1])**(3/2)*f(kai0s))
				#Equation 5
				kai1=(E+1j*parameter_vec[8]*exp(-parameter_vec[9]*((E-E_vec[2])/parameter_vec[8])**2))/E_vec[2]
				kai1s=(E+1j*parameter_vec[8]*exp(-parameter_vec[9]*((E-E_vec[2])/parameter_vec[8])**2))/E_vec[3]
				epsilon_II=-parameter_vec[4]*kai1**(-2)*log(1-kai1**2)-parameter_vec[5]*kai1s**(-2)*log(1-kai1s**2)
				#Equation 8
				epsilon_III=0
				increment=1
				m=1
				while abs(increment) > 1e-4:
					increment = 1/(2*m-1)**3*(parameter_vec[6]/(E_vec[2]-E-1j*parameter_vec[8]*exp(-parameter_vec[9]*((E-E_vec[2])/parameter_vec[8])**2))+parameter_vec[7]/(E_vec[3]-E-1j*parameter_vec[8]*exp(-parameter_vec[9]*((E-E_vec[2])/parameter_vec[8])**2)))
					epsilon_III=epsilon_III+increment
					m=m+1
				#Equation 9
				epsilon_IV=parameter_vec[10]**2/(parameter_vec[13]**2-E**2-1j*E*parameter_vec[11]*exp(-parameter_vec[12]*((E-parameter_vec[13])/parameter_vec[11])**2))+parameter_vec[14]**2/(parameter_vec[17]**2-E**2-1j*E*parameter_vec[15]*exp(-parameter_vec[16]*((E-parameter_vec[17])/parameter_vec[15])**2))+parameter_vec[18]**2/(parameter_vec[21]**2-E**2-1j*E*parameter_vec[19]*exp(-parameter_vec[20]*((E-parameter_vec[21])/parameter_vec[19])**2))
				#Equation 11
				epsilon=sqrt(parameter_vec[0]+epsilon_I+epsilon_II+epsilon_III+epsilon_IV)
				nr = epsilon.real
				ni = 0
		elif model == "Iu_2022": #From Meng Iu [Helmy group] 2022
			nr, ni = Refractive_Index(wl, "AlGaInAsQW_{"+str(x)+"}", "Djurisic")
			c = 3e8 #approximate speed of light [m/s] (required to be approx by model)
			nr = nr+(0.3243*exp(0.6814*c/wl/1e14)+1.609e-13*exp(7.609*c/wl/1e14))/1000*50
			if wl < 900e-9:
				nr = nr-0.144119 #additional offset to account for old model using Djurisic for 780 and Gehrsitz for 1550
		else:
			raise Exception("Refractive index model "+model+" for "+material+" not found")
	elif material[:15] == "AlGaAsP_Barrier":
		#Use Gehrsitz AlGaAs model for AlGaAsP Barriers
		x = float(material.split("{")[1].split("}")[0])
		nr, ni = Refractive_Index(wl, "AlGaAs_{"+str(x)+"}", "Gehrsitz")
	elif material[:6] == "AlGaAs":
		x = float(material.split("{")[1].split("}")[0])
		if model == "Gehrsitz" or model == "Default": #From Gehrsitz 2000
			T = 293
			nr = sqrt(5.9613+7.178e-4*T-0.953e-6*T**2-16.159*x+43.511*x**2-71.317*x**3+57.535*x**4-17.451*x**5+1/(50.535-150.7*x-62.209*x**2+797.16*x**3-1125*x**4+503.79*x**5)/((1.225316977778989+0.023083578135884*(1-1/tanh(92.255357763322920/T))+0.029810239269821*(1-1/tanh(194.9547182923050/T))+1.1308*x+0.1436*x**2)**2-(1e-6/wl)**2)+(21.5647+113.74*x-122.5*x**2+108.401*x**3-47.318*x**4)/((4.7171-3.237e-4*T-1.358e-6*T**2+11.006*x-3.08*x**2)-(1e-6/wl)**2)+(1-x)*1.55e-3/(0.724*1e-3-(1e-6/wl)**2)+x*2.61e-3/(1.331*1e-3-(1e-6/wl)**2))
			ni = 0
		elif model == "Iu_2022": #From Meng Iu [Helmy group] 2022
			nr, ni = Refractive_Index(wl, "AlGaAs_{"+str(x)+"}", "Gehrsitz")
			c = 3e8 #approximate speed of light [m/s] (required to be approx by model)
			if x > 0.45:
				correction = (1.72*exp(0.03609*(c/wl/1e14-3.747)/0.1473)+0.2316*exp(0.3428*(c/wl/1e14-3.747)/0.1473))/1000*9
			else:
				correction = (4.704*exp(0.1839*(c/wl/1e14-3.738)/0.152)+0.0321*exp(3.079*(c/wl/1e14-3.738)/0.152))/1000*9
			nr = nr+correction
		else:
			raise Exception("Refractive index model "+model+" for "+material+" not found")
	elif material[:5] == "AlGaN": #From Laws 2001, most accurate for x<0.4
		x = float(material.split("{")[1].split("}")[0])
		nr = sqrt(1+(4.141-x-4.4*x**2)*wl**2/(wl**2-(187.4e-9-121e-9*x)**2))
		ni = 0
	elif material[:7] == "GaAlInN": #From Peng 1996
		x = float(material.split("{")[1].split("}")[0])
		y = float(material.split("{")[2].split("}")[0])
		nr = sqrt((x*y*(13.55+(9.31-13.55)*(1-x+y)/2)+y*(1-x-y)*(53.57+(13.55-53.57)*(2-x-2*y)/2)+x*(1-x-y)*(53.57+(9.31-53.57)*(2-2*x-y)/2))/(x*y+y*(1-x-y)+x*(1-x-y))*(4.136e-15*299792458/((x*y*(6.2+(3.4-6.2-0.5)*(1-x+y)/2+0.5*((1-x+y)/2)**2)+y*(1-x-y)*(1.9+2*((2-x-2*y)/2)**2+2.3*((2-x-2*y)/2)**15)+x*(1-x-y)*(1.9+(3.4-1.9-1)*(2-2*x-y)/2+((2-2*x-y)/2)**2))/(x*y+y*(1-x-y)+x*(1-x-y)))/wl)**-2*(2-sqrt(1+4.136e-15*299792458/((x*y*(6.2+(3.4-6.2-0.5)*(1-x+y)/2+0.5*((1-x+y)/2)**2)+y*(1-x-y)*(1.9+2*((2-x-2*y)/2)**2+2.3*((2-x-2*y)/2)**15)+x*(1-x-y)*(1.9+(3.4-1.9-1)*(2-2*x-y)/2+((2-2*x-y)/2)**2))/(x*y+y*(1-x-y)+x*(1-x-y)))/wl)-sqrt(1-4.136e-15*299792458/((x*y*(6.2+(3.4-6.2-0.5)*(1-x+y)/2+0.5*((1-x+y)/2)**2)+y*(1-x-y)*(1.9+2*((2-x-2*y)/2)**2+2.3*((2-x-2*y)/2)**15)+x*(1-x-y)*(1.9+(3.4-1.9-1)*(2-2*x-y)/2+((2-2*x-y)/2)**2))/(x*y+y*(1-x-y)+x*(1-x-y)))/wl))+(x*y*(2.05+(3.03-2.05)*(1-x+y)/2)+y*(1-x-y)*(-9.19+(2.05+9.19)*(2-x-2*y)/2)+x*(1-x-y)*(-9.19+(3.03+9.19)*(2-2*x-y)/2))/(x*y+y*(1-x-y)+x*(1-x-y)))
		ni = 0
	else:
		raise Exception("Refractive index model "+material+" not found")
	return nr.real, ni

def colours(material):
	#########################################################################
	# Generates a plotting colour of a material                             #
	#                                                                       #
	# Inputs:                                                               #
	# material is a string containing the name of the material of interest  #
	#                                                                       #
	# Outputs:                                                              #
	# colour is an RGB colour vector with elements between 0 and 1          #
	#########################################################################
	# Semiconductor
	if material == "GaAs":
		return colours("AlGaAs_{0}")
	elif len(material)>8 and material[:8] == "AlGaAs_{":
		x = float(material[8:-1])*0.75+0.25
		return [x, x/1.2, x]
	elif len(material)>12 and material[:12] == "AlGaInAsQW_{":
		x = float(material[12:-1])*0.75+0.25
		return [x, x/1.2, x]
	elif len(material)>17 and material[:17] == "AlGaAsP_Barrier_{":
		x = float(material[17:-1])
		return [x, x/1.2, x]
	# Photoresists
	elif material == "AP3000":
		return [0, 0.3, 0]
	elif material == "BCB":
		return [0, 0.4, 0]
	elif material == "BCB_{Cured}":
		return [0.4, 0.4, 0.4]
	elif material == "ZEP":
		return [0, 0, 0.5]
	elif material == "ZEP_{Exposed}":
		return [0, 0, 0.8]
	elif material == "PMGI":
		return [0.5, 0, 0]
	elif material == "PMGI_{Exposed}":
		return [0.8, 0, 0]
	elif material == "S1811":
		return [0.6, 0, 0]
	elif material == "S1811_{Exposed}":
		return [1, 0, 0]
	# Others
	elif material == "SiO_2":
		return [0.3, 0.3, 0.3]
	elif material == "Au":
		return [0.91, 0.41, 0.17]
	elif material == "Air":
		return [1, 1, 1]
	elif material == "Wax":
		return [1, 1, 0]
	else:
		print(material+" not found in function colours()")
		return [1, 1, 1]

def Effective_Nonlinearity(wl, material):
	#########################################################################
	# Finds the effective nonlinearity of a material at a given wavelength  #
	#                                                                       #
	# Inputs:                                                               #
	# wl is the wavelength [m]                                              #
	# material is a string containing the name of the material of interest  #
	#                                                                       #
	# Outputs:                                                              #
	# d_eff is the effective nonlinearity [m/V]                             #
	#########################################################################
	if material == "Air":
		d_eff = 0
	elif material[:6] == "AlGaAs" or material[:10] == "AlGaInAsQW" or material[:15] == "AlGaAsP_Barrier":
		x = float(material.split("{")[1].split("}")[0])
		#GaAs data from Kondo Photonics Based on Wavelength Integration and Manipulation 2005, fit done in Abolghasem"s PhD thesis
		d_eff_GaAs = 86e-12+4058e-12*exp(-wl/194e-9)+4081e-12*exp(-wl/194e-9)+152e-12*exp(-wl/993e-9)
		#AlGaAs data from Ohashi  JAP 1993, fit done in Abolghasem"s PhD thesis
		d_eff = (-1.9230*x**5+6.4690*x**4-6.0110*x**3+0.8700*x**2-0.1758*x+1.0010)*d_eff_GaAs
	else:
		raise Exception("Effective nonlinearity model "+material+" not found")
	return d_eff

def Nonlinear_Efficiency(lambda_s, lambda_i, n_eff_p, n_eff_s, n_eff_i, E_p, E_s, E_i, x, alpha_p, alpha_s, alpha_i, d, d_eff, delta_k, L):
	#########################################################################
	# Finds the nonlinear conversion efficiency using Eqns. 2.31-2.35 of    #
	# Payam Abolghasem's PhD thesis.                                        #
	#                                                                       #
	# Inputs:                                                               #
	# p, s, and i refer to the pump, signal, and idler respectively         #
	# wl is the wavelength [m]                                              #
	# n_eff is the effective refractive index of the mode                   #
	# E is the normalized electric field as a function of x [m]             #
	# alpha is the propagation loss [1/m]                                   #
	# d is a vector containing the thicknesses of each layer [m]            #
	# d_eff is a vector containing the nonlinear coefficients [m/V]         #
	# delta_k is the phase mismatch [1/m]                                   #
	# L is the sample length [m]                                            #
	#                                                                       #
	# Outputs:                                                              #
	# eta is the nonlinear efficiency per unit width [m/W]                  #
	# eta_norm is the normalized nonlinear efficiency per unit width [1/Wm] #
	#########################################################################
	#Initialize constants
	mu0 = pi*4e-7         # vacuum permeability [H/m]
	c = 299792458         # speed of light [m/s]
	epsilon0 = 1/mu0/c**2 # vacuum permittivity [F/m]
	#determine the layer number for each x value
	layer = [0]*len(x)
	for i in range(1,len(d)):
		layer = [layer[j]+1 if x[j]>sum(d[:i]) else layer[j] for j in range(len(x))]
	#Find d_eff as a function of x
	d_eff_x_vector = [d_eff[l] for l in layer]
	#convert to numpy arrays for speed
	E_p = np.array(E_p)
	E_s = np.array(E_s)
	E_i = np.array(E_i)
	d_eff_x_vector = np.array(d_eff_x_vector)
	#Find nonlinear parameters
	overlap = np.trapz(E_p.conjugate()*E_s*E_i,x) #Field overlap integral
	d_eff_waveguide  = abs(np.trapz(E_p.conjugate()*E_s*E_i*d_eff_x_vector,x))/abs(overlap) #Effective nonlinear coefficient of waveguide Eq. 2.32 [m/V]
	kappa_s = sqrt((8*pi**2*d_eff_waveguide**2)/(n_eff_p*n_eff_s*n_eff_i*epsilon0*c*lambda_s**2)) #Nonlinear coupling factor for signal Eq. 2.31 [W^(-1/2)]
	kappa_i = sqrt((8*pi**2*d_eff_waveguide**2)/(n_eff_p*n_eff_s*n_eff_i*epsilon0*c*lambda_i**2)) #Nonlinear coupling factor for idler Eq. 2.31 [W^(-1/2)]
	xi = overlap/(sqrt(np.trapz(E_p.conjugate()*E_p,x)*np.trapz(E_s.conjugate()*E_s,x)*np.trapz(E_i.conjugate()*E_i,x))) #Nonlinear spatial overlap factor Eq. 2.33 [m^(-1/2)]
	#Find nonlinear efficiency
	if delta_k == 0 and alpha_p==0 and alpha_s==0 and alpha_i==0:
		eta = kappa_s*kappa_i*xi**2*L**2 #Eq. 2.34 and 2.35 [m/W]
	else:
		eta = kappa_s*kappa_i*xi**2*L**2*exp(-(alpha_p+alpha_s+alpha_i)*L/2)*((sin(delta_k*L/2))**2 + (sinh((alpha_s+alpha_i-alpha_p)*L/4))**2)/((delta_k*L/2)**2 + ((alpha_s+alpha_i-alpha_p)*L/4)) #Eq. 2.34 and 2.35 [m/W]
	#Find normalized nonlinear efficiency
	eta_norm = eta*100/L**2 #Eq. 2.35 [%/Wm]
	return eta, eta_norm

def Matching_Layer_Thickness(wl, n, d, bottom_ML_index, top_ML_index, index_peak, n_eff, polarization, bottom_q=0, top_q=0):
	#########################################################################
	# Finds the thickness of a matching layer to make the field peak in the #
	# desired location.                                                     #
	#                                                                       #
	# Inputs:                                                               #
	# wl is the wavelength [m]                                              #
	# n is a vector containing the refractive index of each layer           #
	# d is a vector containing the thicknesses of each layer [m]            #
	# bottom_ML_index is the layer number of the bottom matching layer      #
	# top_ML_index is the layer number of the top matching layer            #
	# index_peak is the layer number where the field should peak            #
	# n_eff is the effective refractive index of the mode                   #
	# polarization is the polarization of the mode (TE or TM)               #
	# bottom_q is the solution number for the bottom ML (0 is thinnest)     #
	# top_q is the solution number for the top ML (0 is thinnest)           #
	#                                                                       #
	# Outputs:                                                              #
	# d_m_bottom is the thickness of the bottom matching layer [m]          #
	# d_m_top is the thickness of the top matching layer [m]                #
	#########################################################################
	d = [i for i in d] #make copy of d to not edit the original
	#Set parameter for TE/TM modes
	if polarization == "TE":
		rho = 0
	elif polarization == "TM":
		rho = 1
	else:
		raise Exception("Invalid polarization. Please use either TE or TM")
	#calculate k vector
	k = [2*pi/wl*sqrt(ni**2-n_eff**2) for ni in n]
	# half the center layer thickness and use in both calculations
	d[index_peak] = d[index_peak]/2
	## Find bottom matching layer thickness
	#calculate M matrix for core bottom half
	M_core = [[1,0],[0,1]]
	for j in range(bottom_ML_index+1,index_peak+1):
		M_core = matmul2x2([[cos(k[j]*d[j]), -n[j]**(2*rho)/k[j]*sin(k[j]*d[j])],[k[j]/n[j]**(2*rho)*sin(k[j]*d[j]), cos(k[j]*d[j])]],M_core)
	#find approximate matching layer thickness
	if (n[bottom_ML_index-2]**(2*rho)*k[bottom_ML_index-1]).real < (n[bottom_ML_index-1]**(2*rho)*k[bottom_ML_index-2]).real:
		d_m_bottom = 1/k[bottom_ML_index]*atan(-n[bottom_ML_index]**(2*rho)/k[bottom_ML_index]*M_core[1][0]/M_core[1][1])
	else:
		d_m_bottom = 1/k[bottom_ML_index]*atan(1/(n[bottom_ML_index]**(2*rho)/k[bottom_ML_index]*M_core[1][0]/M_core[1][1]))
	#If first solution is negative, increment to next solution
	if d_m_bottom.real < 0:
	   d_m_bottom = d_m_bottom + pi/k[bottom_ML_index]
	d_m_bottom = d_m_bottom + bottom_q*pi/k[bottom_ML_index]
	## Find top matching layer thickness
	#calculate M matrix for core top half
	M_core = [[1,0],[0,1]]
	for j in range(top_ML_index-1,index_peak-1,-1):
		M_core = matmul2x2([[cos(k[j]*d[j]), -n[j]**(2*rho)/k[j]*sin(k[j]*d[j])],[k[j]/n[j]**(2*rho)*sin(k[j]*d[j]), cos(k[j]*d[j])]],M_core)
	#find approximate matching layer thickness
	if (n[top_ML_index+2]**(2*rho)*k[top_ML_index+1]).real < (n[top_ML_index+1]**(2*rho)*k[top_ML_index+2]).real:
		d_m_top = 1/k[top_ML_index]*atan(-n[top_ML_index]**(2*rho)/k[top_ML_index]*M_core[1][0]/M_core[1][1])
	else:
		d_m_top = 1/k[top_ML_index]*atan(1/(n[top_ML_index]**(2*rho)/k[top_ML_index]*M_core[1][0]/M_core[1][1]))
	#If first solution is negative, increment to next solution
	if d_m_top.real < 0:
	   d_m_top = d_m_top + pi/k[top_ML_index]
	d_m_top = d_m_top + top_q*pi/k[top_ML_index]
	return d_m_bottom.real, d_m_top.real

def Find_Mode_Near(wl, n, d, polarization, n_eff_guess, max_iterations=100, max_error=1e-12):
	#########################################################################
	# Uses Newton's method to find the value of n_eff close to              #
	# n_eff_guess that makes the characteristic equation equal zero for a   #
	# given waveguide geometry                                              #
	#                                                                       #
	# Inputs:                                                               #
	# wl is the wavelength [m]                                              #
	# n is a vector containing the refractive index of each layer           #
	# d is a vector containing the thicknesses of each layer [m]            #
	# polarization is the polarization of the mode (TE or TM)               #
	# n_eff_guess is the expected effective refractive index of the mode    #
	# max_iterations is the maximum number of iterations to converge        #
	# max_error is the percent error required to be considered converged    #
	#                                                                       #
	# Outputs:                                                              #
	# n_eff is the effective refractive index of the mode                   #
	#########################################################################
	#Set parameter for TE/TM modes
	if polarization == "TE":
		rho = 0
	elif polarization == "TM":
		rho = 1
	else:
		raise Exception("Invalid polarization. Please use either TE or TM")
	## Find M matrix for structure
	n_eff_guesses = [0 for _ in range(max_iterations)]
	for iteration in range(max_iterations):
		n_eff_guesses[iteration] = n_eff_guess
		#calculate k vector
		n_s = n[0]
		n_t = n[-1]
		k_0 = 2*pi/wl
		k = [k_0*sqrt(ni**2-n_eff_guess**2) for ni in n]
		gamma_s = 1j*k[0]
		gamma_t = 1j*k[-1]
		#Ensure real parts of gamma_s and gamma_t are positive (decaying fields)
		#Don't flip imaginary part or it will not converge
		if gamma_s.real < 0:
			gamma_s = -gamma_s.conjugate()
		if gamma_t.real < 0:
			gamma_t = -gamma_t.conjugate()
		#calculate characteristic equation
		M = [[1,0],[0,1]]
		dM = [[0,0],[0,0]]
		for i in range(len(n)-2,0,-1):
			#M matrix and derivative of layer
			M_i = [[cos(k[i]*d[i]), -n[i]**(2*rho)/k[i]*sin(k[i]*d[i])],[k[i]/n[i]**(2*rho)*sin(k[i]*d[i]), cos(k[i]*d[i])]]
			dM_i = [[k_0**2*n_eff_guess*d[i]/k[i]*sin(k[i]*d[i]), k_0**2*n_eff_guess*d[i]*n[i]**(2*rho)/k[i]**2*cos(k[i]*d[i])-k_0**2*n_eff_guess*n[i]**(2*rho)/k[i]**3*sin(k[i]*d[i])],[-k_0**2*n_eff_guess/k[i]/n[i]**(2*rho)*sin(k[i]*d[i])-k_0**2*n_eff_guess*d[i]/n[i]**(2*rho)*cos(k[i]*d[i]), k_0**2*n_eff_guess*d[i]/k[i]*sin(k[i]*d[i])]]
			#Total M matrix and derivative
			dM1 = matmul2x2(dM_i,M)
			dM2 = matmul2x2(M_i,dM)
			dM = [[dM1[j][i]+dM2[j][i] for i in range(len(dM1[0]))] for j in range(len(dM1))]
			M = matmul2x2(M_i,M)
		#characteristic equation
		CE = M[0][0]*gamma_s/n_s**(2*rho)-M[0][1]*gamma_s/n_s**(2*rho)*gamma_t/n_t**(2*rho)-M[1][0]+M[1][1]*gamma_t/n_t**(2*rho)
		#derivative of characteristic equation
		dCE = gamma_s/n_s**(2*rho)*dM[0][0]+M[0][0]/n_s**(2*rho)*k_0**2*n_eff_guess/gamma_s-gamma_s/n_s**(2*rho)*gamma_t/n_t**(2*rho)*dM[0][1]-M[0][1]*gamma_t/n_t**(2*rho)/n_s**(2*rho)*k_0**2*n_eff_guess/gamma_s-M[0][1]*gamma_s/n_s**(2*rho)/n_t**(2*rho)*k_0**2*n_eff_guess/gamma_t-dM[1][0]+gamma_t/n_t**(2*rho)*dM[1][1]+M[1][1]/n_t**(2*rho)*k_0**2*n_eff_guess/gamma_t
		#calculate new n_eff guess and determine error
		n_eff_new = n_eff_guess-CE/dCE
		err = abs(n_eff_new-n_eff_guess)/abs(n_eff_guess)
		if abs(err) < max_error:
			break
		if iteration == max_iterations:
			iterations = [i for i in range(iteration)]
			fig, ax = plt.subplots()
			plt.title("Characteristic Equation as a function of effective index")
			ax.set_xlabel("n$_{eff}$")
			ax.set_ylabel("Characteristic Equation")
			ax.plot(iterations,[n.real for n in n_eff_guesses],'-b',linewidth=2)
			ax.plot(iterations,[n.imag for n in n_eff_guesses],'--r',linewidth=2)
			plt.show()
			raise Exception("n_eff did not converge for n_eff ~"+str(n_eff_guess)+" in "+str(max_iterations)+" iterations - inner loop.")
		n_eff_guess = n_eff_new
	return n_eff_guess

def Mode_Field(wl, n, d, x, n_eff, polarization):
	#########################################################################
	# Finds the normalized electric field of a mode of a structure.         #
	#                                                                       #
	# Inputs:                                                               #
	# wl is the wavelength [m]                                              #
	# n is a vector containing the refractive index of each layer           #
	# d is a vector containing the thicknesses of each layer [m]            #
	# x is the spatial vector along which to find the electric field [m]    #
	# n_eff is the effective refractive index of the mode                   #
	# polarization is the polarization of the mode (TE or TM)               #
	#                                                                       #
	# Outputs:                                                              #
	# E_field is the normalized electric field as a function of x [m]       #
	#########################################################################
	## Define constants
	mu0 = pi*4e-7         # vacuum permeability [H/m]
	c = 299792458         # speed of light [m/s]
	epsilon0 = 1/mu0/c**2 # vacuum permittivity [F/m]
	## Set parameter for TE/TM modes
	if polarization == "TE":
		rho = 0
	elif polarization == "TM":
		rho = 1
	else:
		raise Exception("Invalid polarization. Please use either TE or TM")
	## Initialize variables
	k = [2*pi/wl*sqrt(ni**2-n_eff**2) for ni in n]
	dx = x[1]-x[0]
	#determine the layer number for each x value
	layer = [0 for _ in x]
	layer_num = 0
	next_interface = d[0]
	for i in range(len(x)):
	    if x[i] > next_interface:
	        layer_num += 1
	        next_interface += d[layer_num]
	    layer[i] = layer_num
	## Find the field
	#initialize variables
	EdE = [[0 for _ in range(len(n))] for _ in range(2)]    #EdE is the vector containing E=A and dE/dx=kB
	#Force the coefficient for exponential gain to be zero in the top layer
	EdE[0][-1] = 1
	EdE[1][-1] = 1j*k[-1]/n[-1]**(2*rho)
	if k[-1].imag < 0:
		EdE[1][-1] = -1*EdE[1][-1]
	#calculate the field coefficients at the other layers
	for i in range(len(n)-2,0,-1):
		EdE[0][i] = cos(k[i]*d[i])*EdE[0][i+1]-n[i]**(2*rho)/k[i]*sin(k[i]*d[i])*EdE[1][i+1]
		EdE[1][i] = k[i]/n[i]**(2*rho)*sin(k[i]*d[i])*EdE[0][i+1]+cos(k[i]*d[i])*EdE[1][i+1]
	#reference bottom layer relative to second layer
	EdE[0][0] = EdE[0][1]
	EdE[1][0] = EdE[1][1]
	#find the field for each x value
	field = [0]*len(x)
	for i in range(int(d[0]//dx)+1):
		field[i] = EdE[0][layer[i]]*cos(k[layer[i]]*(x[i]-d[0]))+EdE[1][layer[i]]*n[layer[i]]**(2*rho)/k[layer[i]]*sin(k[layer[i]]*(x[i]-d[0]))
	for i in range(int(d[0]//dx)+1,len(x)):
		field[i] = EdE[0][layer[i]]*cos(k[layer[i]]*(x[i]-sum(d[:layer[i]])))+EdE[1][layer[i]]*n[layer[i]]**(2*rho)/k[layer[i]]*sin(k[layer[i]]*(x[i]-sum(d[:layer[i]])))
	#if TM, convert H to E
	if polarization == "TM":
		E_field = [n_eff/c/epsilon0/n[layer[i]]**2*field[i] for i in range(len(field))]
	else:
		E_field = [i for i in field]
	#normalize the field
	E_max_abs = 0
	E_max = 0
	for Ei in E_field:
		if abs(Ei)>E_max_abs:
			E_max_abs = abs(Ei)
			E_max = Ei
	E_field = [i/E_max for i in E_field]
	return E_field

def Far_Field_Profile(E, x, theta, wl):
	#########################################################################
	# Finds the normalized far field radiation profile as a function of     #
	# angle theta using Eq. 2.7-28 in Heterostucture Lasers Part A by       #
	# Casey, pg. 75                                                         #
	#                                                                       #
	# Inputs:                                                               #
	# E is the normalized electric field as a function of x [m]             #
	# theta is the observation angle [rad]                                  #
	# wl is the wavelength [m]                                              #
	#                                                                       #
	# Outputs:                                                              #
	# E_theta is the normalized electric field as a function of theta [rad] #
	#########################################################################
	n = 1 #Refractive index. Assume far field radiation is in air
	k = n*2*pi/wl
	#convert to numpy arrays for speed
	x = np.array(x)
	E = np.array(E)
	E_theta = [cos(theta[i])*np.trapz(E*np.exp(1j*k*sin(theta[i])*x),x) for i in range(len(theta))]
	#normalize the field
	E_max_abs = 0
	E_max = 0
	for Ei in E_theta:
		if abs(Ei)>E_max_abs:
			E_max_abs = abs(Ei)
			E_max = Ei
	E_theta = [i/E_max for i in E_theta]
	return E_theta

def matmul2x2(A,B):
	#########################################################################
	# Multiplies together two 2x2 matrices, A and B, and returns the result #
	#########################################################################
	a = A[0][0]*B[0][0]+A[0][1]*B[1][0]
	b = A[0][0]*B[0][1]+A[0][1]*B[1][1]
	c = A[1][0]*B[0][0]+A[1][1]*B[1][0]
	d = A[1][0]*B[0][1]+A[1][1]*B[1][1]
	return [[a,b],[c,d]]