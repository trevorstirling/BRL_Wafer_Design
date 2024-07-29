###########################################################################
# BRL8 Design Code                                                        #
# Solves for matching layer thicknesses to make the following structure   #
# phase matched. Created to reproduce BRL8 designed by Nima Zareian.      #
#                                                                         #
# Left in the model to match Nima:                                        #
# -uses GaAs as top layer instead of air                                  #
# -uses Djurisic model with x=28% for QW (likely an accident)             #
# -adjusts refractive index in barriers and QWs due to carrier injection  #
#                                                                         #
# Author: Trevor Stirling                                                 #
# Date: July 6, 2024                                                      #
###########################################################################

import sys
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#Import functions from parent directory
sys.path.append("..") 
from BRL_Common_Functions import colours, Effective_Nonlinearity, Nonlinear_Efficiency, Refractive_Index, Matching_Layer_Thickness, Find_Mode_Near, Mode_Field, Far_Field_Profile
from BRL_File_Generation import save_BRL_to_file, PICS3D_file_generation, Lumerical_file_generation

## Simulation Options
display_neff_loop_progress = False
generate_PICS3D_files = True
generate_Lumerical_files = True
save_data = True
plot_data = False
display_results = True
## Define Variables
wafer_name = 'BRL8'
quantum_well_model = "Djurisic"
AlGaAs_model = "Gehrsitz"
#wavelength and mode properties
lambda_p  = 751e-9                     #Pump (BRW) wavelength
lambda_s  = lambda_p*2                 #Signal (TIR) wavelength
lambda_i  = 1/(1/lambda_p-1/lambda_s)  #Idler (TIR) wavelength
phase_matching_type = "IIA"            #Type of phase matching (I, IIA, or IIB)
#materials
material_top = "AlGaAs_{0}"                    #Top material
material_cap = "AlGaAs_{0}"                    #Cap material
material_1 = "AlGaAs_{0.25}"                   #Bragg stack material b
material_2 = "AlGaAs_{0.7}"                    #Bragg stack material a (next to core)
material_matching = "AlGaAs_{0.2}"             #Matching Layer
material_A = "AlGaAs_{0.7}"                    #A Layer
material_barrier = "AlGaAs_{0.28}"             #Barrier
material_qw = "AlGaInAsQW_{0.11}"              #Quantum Well
material_substrate = "AlGaAs_{0}"              #Substrate material
#number of layers
N_t = 4                    #Number of Bragg bilayers on top
N_b = 5                    #Number of Bragg bilayers on bottom
N_QW = 2                   #Number of Quantum Wells
#thickness of layers
d_A_undoped = 40e-9        #A Layer (undoped section) thickness
d_A_b = 160e-9             #Bottom A Layer (doped section) thickness
d_b = 10e-9                #Barrier thickness' (between quantum wells)
d_QW = 4.6e-9              #Quantum Well thickness
d_A_t = 160e-9             #Top A Layer (doped section) thickness
d_cap = 150e-9             #Cap Layer thickness
q_b = 1                    #Bottom matching layer mode order
q_t = 1                    #Top matching layer mode order
#nonliearity parameters
L = 1e-3                   #sample length [m]
alpha_p = 0                #pump propagation loss [m^-1]
alpha_s = 0                #signal propagation loss [m^-1]
alpha_i = 0                #idler propagation loss [m^-1]
#material properties
Jth = 1250                 #threshold currrent density, unit [A/cm^2]
tau = 3e-9                 #recombination time [s]
e = 1.602176e-19           #electron charge [C]
#PICS3D parameters
doping_substrate = -3e24   #Substrate doping [m^-3]
doping_n_stack = -2e24     #Bragg n stack doping [m^-3]
doping_n_core = -5e23      #n side of core doping [m^-3]
doping_deep_core = 0       #Core doping [m^-3]
doping_p_core = 5e23       #p side of core doping [m^-3]
doping_p_stack = 3e24      #Bragg p stack doping [m^-3]
doping_cap = 3e24          #Cap doping [m^-3]
doping_top = 0             #Top layer doping [m^-3]
d_grading = 25e-9          #Grading layer thickness
etch_depth = 6e-6          #Etch depth [m]
#simulation parameters
max_error = 1e-14          #Error allowable in calculating effective index
max_iterations = 100       #Maximum number of iterations in calculating effective index
x_scale = 1e6              #Scale factor for x axis from meters
## Define Structure
#Layer Names Vector
barrier_layers = ["Barrier"]
for i in range(1,N_QW+1):
	barrier_layers = [*barrier_layers,"QW"+str(i),"Barrier"]
layer_names = ["Substrate",                                              # Substrate
			 *["Bragg1", "Bragg2"]*N_b,                                  # Bragg stack
			 "MatchingLayer",                                            # Matching Layer
			 "ALayer", "ALayer",                                         # A Layer (doped and undoped)
			 *barrier_layers,                                            # Barrier/QW structure
			 "ALayer", "ALayer",                                         # A Layer (doped and undoped)
			 "MatchingLayer",                                            # Matching Layer
			 *["Bragg2", "Bragg1"]*N_t,                                  # Bragg stack
			 "Cap",                                                      # Cap Layer
			 "Top"]                                                      # Top Layer
#Material Vector
materials = [material_substrate,                                         # Substrate
			 *[material_1, material_2]*N_b,                              # Bragg stack
			 material_matching,                                          # Matching Layer
			 material_A, material_A,                                     # A Layer (doped and undoped)
			 *[material_barrier, material_qw]*N_QW, material_barrier,    # Barrier/QW structure
			 material_A, material_A,                                     # A Layer (doped and undoped)
			 material_matching,                                          # Matching Layer
			 *[material_2, material_1]*N_t,                              # Bragg stack
			 material_cap,                                               # Cap Layer
			 material_top]                                               # Top Layer
#Doping Vector
doping = [doping_substrate,                                              # Substrate
		  *[doping_n_stack, doping_n_stack]*N_b,                         # Bragg stack
		  doping_n_core,                                                 # Matching Layer
		  doping_n_core, doping_deep_core,                               # A Layer (doped and undoped)
		  *[doping_deep_core, doping_deep_core]*N_QW, doping_deep_core,  # Barrier/QW structure
		  doping_deep_core, doping_p_core,                               # A Layer (doped and undoped)
		  doping_p_core,                                                 # Matching Layer
		  *[doping_p_stack, doping_p_stack]*N_t,                         # Bragg stack
		  doping_cap,                                                    # Cap Layer
		  doping_top]                                                    # Top Layer
#Find indices pointing at various layers
index_s = 0                                                                            # Substrate
A_indices = [i for i in range(len(layer_names)) if layer_names[i] == "ALayer"]
index_A_b = A_indices[0]                                                               # Bottom A Layer (doped)
index_A_undoped = A_indices[-3:-1]                                                     # A Layer (undoped)
index_A_t = A_indices[-1]                                                              # Top A Layer (doped)
ML_indices = [i for i in range(len(layer_names)) if layer_names[i] == "MatchingLayer"]
index_matching_b = ML_indices[0]                                                       # Bottom Matching Layer
index_matching_t = ML_indices[1]                                                       # Top Matching Layer
index_b = [i for i in range(len(layer_names)) if layer_names[i] == "Barrier"]          # Barriers
index_QW = [i for i in range(len(layer_names)) if layer_names[i][:2] == "QW"]          # Quantum Wells
index_cap = len(materials)-2                                                           # Cap Layer
index_t = len(materials)-1                                                             # Top Layer
#Find index pointing at layer where Bragg mode should peak
if N_QW%2 == 1:
	index_peak = index_QW[N_QW//2] #middle quantum well
else:
	index_peak = index_QW[N_QW//2]-1 #between two middle quantum wells
#ensure signal wavelength is less than idler
if lambda_s > lambda_i:
	lambda_temp = lambda_s
	lambda_s = lambda_i
	lambda_i = lambda_temp
#get refractive indices, nonlinearity, and colours (for plotting) of materials
models = [AlGaAs_model for _ in materials]
for i in index_QW:
	models[i] = quantum_well_model
n_p = [Refractive_Index(lambda_p, materials[i], models[i])[0] for i in range(len(materials))]
n_s = [Refractive_Index(lambda_s, materials[i], models[i])[0] for i in range(len(materials))]
n_i = [Refractive_Index(lambda_i, materials[i], models[i])[0] for i in range(len(materials))]
d_eff_vector = [Effective_Nonlinearity(lambda_p,m) for m in materials]
colour_vector = [colours(m) for m in materials]
#Adjust refractive indices of core due to carrier injection
Nc = Jth*tau/((N_QW*d_QW+(N_QW+1)*d_b)*100*e) #Injected carriers at threshold [cm^-3]
d_n = [-1e-20*Nc if i in [*index_QW,*index_b] else 0 for i in range(len(materials))]
#Nima's code accidentally used 28% in the QW at the pump wavelength, so that has been recreated here
for i in index_QW:
	d_n[i] = d_n[i] + Refractive_Index(lambda_p, "AlGaInAsQW_{0.28}",quantum_well_model)[0]-Refractive_Index(lambda_p, "AlGaInAsQW_{0.11}",quantum_well_model)[0]
n_p = [n_p[i]+d_n[i] for i in range(len(n_p))]
#Determine polarization of modes based on phase matching type
if phase_matching_type == "I":
	polarization_p = 'TM'
	polarization_s = 'TE'
	polarization_i = 'TE'
elif phase_matching_type == "IIA":
	polarization_p = 'TE'
	polarization_s = 'TM'
	polarization_i = 'TE'
elif phase_matching_type == "IIB":
	polarization_p = 'TE'
	polarization_s = 'TE'
	polarization_i = 'TM'
else:
	raise Exception("Phase matching type must be either I, IIA, or IIB")
#Set units for x axis
if x_scale == 1e9:
	x_unit = 'nm'
elif x_scale == 1e6:
	x_unit = 'µm'
else:
	x_unit = 'undef.'
## Loop through possible n_eff, find Matching Layer thickness that allows a Bragg mode, and find TIR index for that structure
#create n_eff vector of possible values greater than the lowest TIR (signal or idler) index, and less than the lowest BRW (pump) index
n_p_min = min([*n_s[1:-1],*n_i[1:-1]])
n_p_max = min(n_p[1:-1])
num_points = 500
n_eff_p = [n_p_min+(n_p_max-n_p_min)/(num_points+1)*(i+1) for i in range(num_points)] #linear spacing not including start or end point
#initialize loop variables
n_eff_s = [0]*len(n_eff_p)
n_eff_i = [0]*len(n_eff_p)
for i in range(len(n_eff_p)):
	#calculate approximate k vector
	k_p_guess = [2*pi/lambda_p*sqrt(n**2-n_eff_p[i]**2) if n > n_eff_p[i] else 1 for n in n_p]
	#calculate approximate d vector
	d_vector = [pi/2/k for k in k_p_guess]
	d_vector[index_s] = 500e-9
	for j in index_A_undoped:
		d_vector[j] = d_A_undoped
	d_vector[index_A_b] = d_A_b
	for j in index_b:
		d_vector[j] = d_b
	for j in index_QW:
		d_vector[j] = d_QW
	d_vector[index_A_t] = d_A_t
	d_vector[index_cap] = d_cap
	d_vector[index_t] = 500e-9
	#calculate matching layer thicnkesses
	d_vector[index_matching_b], d_vector[index_matching_t] = Matching_Layer_Thickness(lambda_p, n_p, d_vector, index_matching_b, index_matching_t, index_peak, n_eff_p[i], polarization_p, q_b, q_t)
	#Find TIR modes for this structure
	if i == 0:
		n_eff_s[-1] = (max(n_s)+min(n_s[1:-1]))/2
		n_eff_i[-1] = (max(n_i)+min(n_i[1:-1]))/2
	n_eff_s[i] = Find_Mode_Near(lambda_s, n_s, d_vector, polarization_s, n_eff_s[i-1], max_iterations, max_error)
	n_eff_i[i] = Find_Mode_Near(lambda_i, n_i, d_vector, polarization_i, n_eff_i[i-1], max_iterations, max_error)
	if display_neff_loop_progress:
		if i%100 == 0:
			print("{:.2f}% complete n_eff loop".format((i)/len(n_eff_p)*100))
		if i == len(n_eff_p):
			print("")
## Find n_eff where BRW and TIR modes are phase matched (delta_k = 0)
delta_k = [2*pi*(n_eff_p[i].real/lambda_p-n_eff_i[i].real/lambda_i-n_eff_s[i].real/lambda_s) for i in range(len(n_eff_p))]
#Plot effective indices as a function BRW effective index
if plot_data:
	fig, ax = plt.subplots()
	plt.title("Effective indices as a function BRW effective index")
	ax.set_xlabel("Pump (BRW) n$_{eff}$ [nm]")
	ax.set_ylabel("n$_{eff}$")
	ax.plot(n_eff_p,n_eff_p,linewidth=2)
	ax.plot(n_eff_p,[n.real for n in n_eff_s],linewidth=2)
	ax.plot(n_eff_p,[n.real for n in n_eff_i],linewidth=2)
	ax.legend(["Pump (BRW)", "Signal (TIR$_{"+polarization_s+"}$)", "Idler (TIR$_{"+polarization_i+"}$)"],loc='upper left')
	#Plot delta k as a function BRW effective index
	fig, ax = plt.subplots()
	plt.title("Phase matching ∆k as a function BRW effective index")
	ax.set_xlabel("Pump (BRW) n$_{eff}$ [nm]")
	ax.set_ylabel("∆k")
	ax.plot(n_eff_p,delta_k,linewidth=2)
	ax.plot([n_eff_p[0],n_eff_p[-1]],[0,0],'--k',linewidth=2)
crossing = [i for i in range(1,len(n_eff_p)) if delta_k[i]*delta_k[i-1] < 0]
if len(crossing) == 1:
	crossing = crossing[0]
else:
	if plot_data:
		plt.show()
	raise Exception('Phase matching is not met only once for this range of n_eff_p')
#interpolate n_eff where phase matching occurs
n_eff_p_matched = n_eff_p[crossing-1]-delta_k[crossing-1]*(n_eff_p[crossing]-n_eff_p[crossing-1])/(delta_k[crossing]-delta_k[crossing-1])
n_eff_p_matched = n_eff_p_matched.real #Ignore losses for BRW index to avoid complex layer widths
## Find structure of this n_eff_p
k_p = [2*pi/lambda_p*sqrt(n**2-n_eff_p_matched**2) if n > n_eff_p_matched else 1 for n in n_p]
#calculate d vector
d_vector = [pi/2/k for k in k_p]
d_vector[index_s] = 500e-9
for j in index_A_undoped:
	d_vector[j] = d_A_undoped
d_vector[index_A_b] = d_A_b
for j in index_b:
	d_vector[j] = d_b
for j in index_QW:
	d_vector[j] = d_QW
d_vector[index_A_t] = d_A_t
d_vector[index_cap] = d_cap
d_vector[index_t] = 500e-9
#calculate matching layer thicnkesses
d_vector[index_matching_b], d_vector[index_matching_t] = Matching_Layer_Thickness(lambda_p, n_p, d_vector, index_matching_b, index_matching_t, index_peak, n_eff_p_matched, polarization_p, q_b, q_t);
#calculate actual effective indices and phase mismatch
n_eff_s_matched = Find_Mode_Near(lambda_s, n_s, d_vector, "TM", n_eff_s[crossing-1], max_iterations, max_error)
n_eff_i_matched = Find_Mode_Near(lambda_i, n_i, d_vector, "TE", n_eff_i[crossing-1], max_iterations, max_error)
n_eff_p_matched = Find_Mode_Near(lambda_p, n_p, d_vector, polarization_p, n_eff_p_matched, max_iterations, max_error)
delta_k_matched = 2*pi*(n_eff_p_matched.real/lambda_p-n_eff_i_matched.real/lambda_i-n_eff_s_matched.real/lambda_s)
## Prepare x vector for the field
#create x vector with at least 100 points per material
dx = min(d_vector[1:-1])/100
x = [i*dx for i in range(int(sum(d_vector)/dx))]
#determine the layer number for each x value
layer = [0 for _ in x]
layer_num = 0
next_interface = d_vector[0]
for i in range(len(x)):
    if x[i] > next_interface:
        layer_num += 1
        next_interface += d_vector[layer_num]
    layer[i] = layer_num
## Find the refractive index as a function of x
nx_p = [n_p[l] for l in layer]
nx_s = [n_s[l] for l in layer]
nx_i = [n_i[l] for l in layer]
nx_min = min([*n_p[1:-1],*n_s[1:-1],*n_i[1:-1]])*0.9
nx_max = max([*n_p,*n_s,*n_i])*1.1
## Find Fields
E_p = Mode_Field(lambda_p, n_p, d_vector, x, n_eff_p_matched, polarization_p)
E_s = Mode_Field(lambda_s, n_s, d_vector, x, n_eff_s_matched, polarization_s)
E_i = Mode_Field(lambda_i, n_i, d_vector, x, n_eff_i_matched, polarization_i)
## Plot the far field profile
if plot_data:
	num_theta_points = 100
	theta = [-pi/2+i*pi/(num_theta_points-1) for i in range(num_theta_points)]
	E_p_theta = Far_Field_Profile(E_p,x,theta,lambda_p)
	E_s_theta = Far_Field_Profile(E_s,x,theta,lambda_p)
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	plt.title("Normalized Far Field Intensity [A.U.]")
	ax.set_rlabel_position(45)
	ax.plot(theta,[abs(i)**2 for i in E_p_theta],'k',linewidth=2)
	ax.plot(theta,[abs(i)**2 for i in E_s_theta],'--k',linewidth=2)
	ax.legend(["Pump (BRW)", "Signal (TIR$_{"+polarization_s+"}$)"], loc="upper left")
## Plot the structure with the field
x_zero = d_vector[index_s] #set the zero point to the substrate interface
x_start = [0-x_zero for _ in d_vector]
x_plot = [(xi-x_zero)*x_scale for xi in x]
for i in range(1,len(d_vector)):
	x_start[i] = x_start[i-1]+d_vector[i-1]
if plot_data:
	fig = plt.figure(figsize=(12,8))
	#Pump Refractive Index
	ax1 = plt.subplot(2,3,1)
	ax1.plot(x_plot,nx_p,'k',linewidth=2)
	ax1.plot([x_plot[0],x_plot[-1]],[n_eff_p_matched.real]*2,'--k',linewidth=2)
	ax1.title.set_text("Pump (BRW) Refractive Index Profile")
	#Signal Refractive Index
	ax2 = plt.subplot(2,3,2)
	ax2.plot(x_plot,nx_s,'k',linewidth=2)
	ax2.plot([x_plot[0],x_plot[-1]],[n_eff_s_matched.real]*2,'--k',linewidth=2)
	ax2.title.set_text("Signal (TIR$_{"+polarization_s+"}$) Refractive Index Profile")
	#Idler Refractive Index
	ax3 = plt.subplot(2,3,3)
	ax3.plot(x_plot,nx_i,'k',linewidth=2)
	ax3.plot([x_plot[0],x_plot[-1]],[n_eff_i_matched.real]*2,'--k',linewidth=2)
	ax3.title.set_text("Idler (TIR$_{"+polarization_i+"}$) Refractive Index Profile")
	for ax in [ax1,ax2,ax3]:
		ax.set_xlabel("Distance From Substrate ["+x_unit+"]")
		ax.set_ylabel('n')
		ax.set_xlim([0-x_zero*x_scale,(sum(d_vector)-x_zero)*x_scale])
		ax.set_ylim([nx_min,nx_max])
	#Structure with Fields
	ax4 = plt.subplot(2,1,2)
	y_min = min([*[i.real for i in E_p],*[i.real for i in E_i],*[i.real for i in E_s]])
	y_max = max([*[i.real for i in E_p],*[i.real for i in E_i],*[i.real for i in E_s]])
	for i in range(len(d_vector)):
		ax4.add_patch(patches.Rectangle([x_start[i]*x_scale,y_min], d_vector[i]*x_scale, y_max-y_min, facecolor=tuple(colour_vector[i]), linestyle="None"))
	ax4.plot(x_plot,[i.real for i in E_p],'b',linewidth=2)
	ax4.plot(x_plot,[i.real for i in E_s],'k',linewidth=2)
	ax4.plot(x_plot,[i.real for i in E_i],'r',linewidth=2)
	ax4.plot([x_plot[0],x_plot[-1]],[0]*2,':k')
	ax4.title.set_text("Normalized Electric Field")
	ax4.set_xlabel("Distance From Substrate ["+x_unit+"]")
	ax4.set_ylabel('Field [A.U.]')
	ax4.set_xlim([0-x_zero*x_scale,(sum(d_vector)-x_zero)*x_scale])
	ax4.set_ylim(y_min,y_max)
	plt.subplots_adjust(hspace=0.3)
## Find the effective nonliearity of the structure
eta, eta_norm = Nonlinear_Efficiency(lambda_s, lambda_i, n_eff_p_matched, n_eff_s_matched, n_eff_i_matched, E_p, E_s, E_i, x, alpha_p, alpha_s, alpha_i, d_vector, d_eff_vector, delta_k_matched, L)
## Find the overlap factor
QW_intensity_sum = sum([E_p[i].real**2 if layer[i] in index_QW else 0 for i in range(len(E_p))]) #total field intensity in the quantum wells
Gamma_p = QW_intensity_sum/sum([i.real**2 for i in E_p]) #ratio of the field in the Quantum Wells to the total field
## Generate PICS3D Crosslight Simulation Files
if generate_PICS3D_files:
	crosslight_materials = [i for i in materials]
	for i in index_QW:
		crosslight_materials[i] = "AlGaAs_{0.05}" #Change QW mole fraction to 5% for Crosslight
	PICS3D_file_generation(wafer_name, crosslight_materials, d_vector, doping, d_grading, index_A_b, index_A_t, index_QW, index_peak, etch_depth)
## Generate Lumerical Files
if generate_Lumerical_files:
	Lumerical_file_generation(wafer_name, materials, d_vector, models, layer_names, d_n)
## Save structure for later use
if save_data:
	save_BRL_to_file(wafer_name, d_vector, materials, doping)
## Print important quantities to command window
if display_results:
	for i in range(len(n_p)-1,-1,-1):
		print("Layer {:2d}".format(i)+": {:7.3f} nm ".format(d_vector[i]*1e9)+materials[i])
	print("")
	print("The Pump effective index is {:.5g}".format(n_eff_p_matched))
	print("The Signal effective index is {:.5g}".format(n_eff_s_matched))
	print("The Idler effective index is {:.5g}".format(n_eff_i_matched))
	print("")
	print("The magnitude of the BRW wave in the substrate is {:.2g}% of the peak".format(max([i.real for i in E_p[:-int(-d_vector[0]//dx)]])*100))
	print("The phase mismatch is ∆k = {:.3g} ".format(delta_k_matched)+"which is {:.3g}% of k_pump".format(delta_k_matched/(2*pi*n_eff_p_matched.real/lambda_p)))
	print("The epitaxial thickness is {:.4g} µm".format(sum(d_vector[1:-1])*1e6))
	print("The normalized nonlinear conversion efficiency is {:.3g}%m/Wcm^2".format(eta_norm.real/100**2))
	print("The QW overlap factor Gamma p = {:.4g}%".format(Gamma_p*100))
if plot_data:
	plt.show()