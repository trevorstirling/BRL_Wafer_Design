#########################################################################
# BRL File Generating Functions                                         #
# - save_BRL_to_file                                                    #
# - Lumerical_file_generation                                           #
# - PICS3D_file_generation                                              #
#                                                                       #
# Author: Trevor Stirling                                               #
# Date: July 6, 2024                                                    #
#########################################################################

import os
from datetime import datetime
import numpy as np
from BRL_Common_Functions import colours, Refractive_Index

output_dir = os.path.join(os.path.dirname(__file__),'Generated Files')

def get_pics3d_defaults(wafer_name):
	#########################################################################
	# The Arnoldi solver in PICS3D is tempormental and will not always find #
	# the Bragg mode. This function gives parameters which work for wafers  #
	# that have already been tested.                                        #
	#                                                                       #
	#returns x1, y1, x2, y2, num_modes, index                               #
	#########################################################################
	if wafer_name == "BRL7_2D":
		return ["left_edge","etch_interface","ridge_SiO2_col","Ridge_cap_interface","1","3.166"]
	elif wafer_name == "BRL8_2D" or wafer_name == "BRL9_2D":
		return ["left_edge","Substrate_top_interface","ridge_SiO2_col","Cap_contact_interface","1","3.137"]
	elif wafer_name == "BRL10_2D":
		return ["left_edge","Substrate_top_interface","ridge_SiO2_col","Cap_contact_interface","1","3.124"]
	elif wafer_name == "BRL11_2D":
		return ["left_edge","Substrate_top_interface","right_edge","Ridge_cap_interface","1","3.1592"]
	elif wafer_name == "BA0_2D" or wafer_name == "BA1_2D":
		return ["left_edge","Substrate_top_interface","right_edge","Ridge_cap_interface","1","3.11"]
	elif wafer_name == "BA2_2D":
		return ["left_edge","etch_interface","right_edge","Cap_contact_interface","1","3.1633"]
	else:
		return ["left_edge","Substrate_top_interface","ridge_SiO2_col","Cap_contact_interface","30","3.2"]

def check_or_make_directory(dir_path):
	#########################################################################
	# Checks if a director exists, and if not, creates it                   #
	#########################################################################
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path)
		print("Created new directory:", dir_path)

def save_BRL_to_file(wafer_name, d_vector, materials, doping):
	#########################################################################
	# Saves BRL layer thicknesses and materials to a text file              #
	#                                                                       #
	# Inputs:                                                               #
	# wafer_name is the name of the wafer                                   #
	# d is a vector containing the thicknesses of each layer [m]            #
	# materials is a vector containing the names of each layer              #
	#########################################################################
	structures_dir = os.path.join(output_dir,'Structures')
	check_or_make_directory(structures_dir)
	file_name = os.path.join(structures_dir,wafer_name+'_structure.txt')
	with open(file_name, "w") as file:
		file.write("Layer Thickness [nm], Material, Doping [cm^-3]")
		for i in range(len(d_vector)):
			file.write("\n{:7.3f}, ".format(d_vector[i]*1e9)+materials[i]+", {:.2e}".format(doping[i]))
	print("Created "+file_name)

def layer_line_regular(material,p_doping):
	#########################################################################
	# Returns a string in PICS3D format with material info for a layer      #
	#                                                                       #
	# Inputs:                                                               #
	# material is the name of the layer material                            #
	# p_doping is the level of p doping in the layer [m^-3]                 #
	#########################################################################
	x = material.split("{")[1].split("}")[0]
	if p_doping < 0:
		return "layer_mater mater_lib=AlGaAs var_symbol1=x &&\ncolumn_num=1 var1="+x+" n_doping="+str(-p_doping)+"\n"
	elif p_doping > 0:
		return "layer_mater mater_lib=AlGaAs var_symbol1=x &&\ncolumn_num=1 var1="+x+" p_doping="+str(p_doping)+"\n"
	else:
		return "layer_mater mater_lib=AlGaAs var_symbol1=x &&\ncolumn_num=1 var1="+x+"\n"

def layer_line_graded(material1,material2,p_doping):
	#########################################################################
	# Returns a string in PICS3D format with material info for a grading    #
	# layer                                                                 #
	#                                                                       #
	# Inputs:                                                               #
	# material1 is the name of the material before the grading layer        #
	# material2 is the name of the material after the grading layer         #
	# p_doping is the level of p doping in the grading layer [m^-3]         #
	#########################################################################
	x1 = material1.split("{")[1].split("}")[0]
	x2 = material2.split("{")[1].split("}")[0]
	if p_doping < 0:
		return "layer_mater mater_lib=AlGaAs var_symbol1=x grade_var=1 &&\ncolumn_num=1 grade_from="+x1+" grade_to="+x2+" n_doping="+str(-p_doping)+"\n"
	else:
		return "layer_mater mater_lib=AlGaAs var_symbol1=x grade_var=1 &&\ncolumn_num=1 grade_from="+x1+" grade_to="+x2+" p_doping="+str(p_doping)+"\n"

def layer_line_quantum_well(material):
	#########################################################################
	# Returns a string in PICS3D format with material info for a quantum    #
	# well                                                                  #
	#                                                                       #
	# Inputs:                                                               #
	# material is the name of the quantum well material                     #
	#########################################################################
	return "layer_mater mater_lib=AlGaAs var_symbol1=x model=quantum_well &&\ncolumn_num=1 var1="+material.split("{")[1].split("}")[0]+" material_label=QW\n"

def add_layer(file,layer_line,t,depth,etch_depth,t_SiO2,n,layer_num):
	#########################################################################
	# Adds a layer to the PICS3D file accounting for etch depth             #
	#                                                                       #
	# Inputs:                                                               #
	# file is the currently open file to write to                           #
	# layer_line is a string containing the material info for the layer     #
	# t is the layer thickness [µm]                                         #
	# depth is the depth below the surface the layer starts [µm]            #
	# etch_depth is the desired etch depth of the laser [µm]                #
	# t_SiO2 is the thickness of SiO2 covering the laser and sidewall       #
	# n is the number of vertical mesh points to include in the layer       #
	# layer_num is the layer number used for labelling only                 #
	#########################################################################
	min_layer_thickness=0.001 #[µm]
	if depth-t >= etch_depth-min_layer_thickness:
		#Layer completely unetched
		file.write("$ Layer "+str(layer_num)+"\n")
		file.write(layer_line)
		file.write(layer_line.replace("column_num=1","column_num=2"))
		file.write(layer_line.replace("column_num=1","column_num=3"))
		file.write("layer d={:.6g}".format(t)+" n={:d} r=-1.1\n".format(n))
	elif depth-t >= etch_depth-t_SiO2-min_layer_thickness:
		#Top of layer within SiO2
		if depth <= etch_depth+min_layer_thickness:
			#Layer completely within SiO2
			file.write("$ Layer "+str(layer_num)+"\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=SiO2 column_num=3 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer d={:.6g}".format(t)+" n={:d} r=-1.1\n".format(n))
			if abs(depth-etch_depth) <= min_layer_thickness:
				#Bottom of layer is coincident with etch depth
				file.write("layer_position label=etch_interface location=bottom\n")
		else:
			#Bottom of layer unetched
			#AlGaAs
			file.write("$ Layer "+str(layer_num)+" - part 1\n")
			file.write(layer_line)
			file.write(layer_line.replace("column_num=1","column_num=2"))
			file.write(layer_line.replace("column_num=1","column_num=3"))
			file.write("layer d={:.6g}".format(depth-etch_depth)+" n={:d} r=-1.1\n".format(n))
			#SiO2
			file.write("$ Layer "+str(layer_num)+" - part 2\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=SiO2 column_num=3 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer d={:.6g}".format(t-depth+etch_depth)+" n={:d} r=-1.1\n".format(n))
			file.write("layer_position label=etch_interface location=bottom\n")
		if abs(depth-t-etch_depth+t_SiO2) <= min_layer_thickness:
			#Top of layer is coincident with top of SiO2
			file.write("layer_position  label=SiO2_air_interface  location=top\n")
	else:
		#Top of layer above SiO2
		if depth <= etch_depth-t_SiO2+min_layer_thickness:
			#Layer completely within Air
			file.write("$ Layer "+str(layer_num)+"\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=Air column_num=3 insulator_macro=yes\n")
			file.write("layer d={:.6g}".format(t)+" n={:d} r=-1.1\n".format(n))
		elif depth <= etch_depth+min_layer_thickness:
			#Bottom of layer within SiO2
			#SiO2
			file.write("$ Layer "+str(layer_num)+" - part 1\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=SiO2 column_num=3 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer d={:.6g}".format(depth-etch_depth+t_SiO2)+" n={:d} r=-1.1\n".format(n))
			if abs(depth-etch_depth) <= min_layer_thickness:
				#Bottom of layer is coincident with etch depth
				file.write("layer_position label=etch_interface location=bottom\n")
			file.write("layer_position label=SiO2_air_interface location=top\n")
			#Air
			file.write("$ Layer "+str(layer_num)+" - part 2\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=Air column_num=3 insulator_macro=yes\n")
			file.write("layer d={:.6g}".format(t-depth+etch_depth-t_SiO2)+" n={:d} r=-1.1\n".format(n))
		else:
			#Bottom of layer unetched
			#AlGaAs
			file.write("$ Layer "+str(layer_num)+" - part 1\n")
			file.write(layer_line)
			file.write(layer_line.replace("column_num=1","column_num=2"))
			file.write(layer_line.replace("column_num=1","column_num=3"))
			file.write("layer d={:.6g}".format(depth-etch_depth)+" n={:d} r=-1.1\n".format(n))
			#SiO2
			file.write("$ Layer "+str(layer_num)+" - part 2\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=SiO2 column_num=3 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer d={:.6g}".format(t_SiO2)+" n={:d} r=-1.1\n".format(n))
			file.write("layer_position label=etch_interface location=bottom\n")
			file.write("layer_position label=SiO2_air_interface location=top\n")
			#Air
			file.write("$ Layer "+str(layer_num)+" - part 3\n")
			file.write(layer_line)
			file.write("layer_mater mater_lib=SiO2 column_num=2 insulator_macro=yes &&\n")
			file.write("material_label=insulator\n")
			file.write("layer_mater mater_lib=Air column_num=3 insulator_macro=yes\n")
			file.write("layer d={:.6g}".format(t-t_SiO2-depth+etch_depth)+" n={:d} r=-1.1\n".format(n))

def PICS3D_file_generation(wafer_name, materials, d_vector, doping, d_grading, index_A_b, index_A_t, index_QW, index_peak, etch_depth):
	#########################################################################
	# Creates files to run PICS3D Crosslight Simulations                    #
	#                                                                       #
	# Inputs:                                                               #
	# wafer_name is a string containing the name of the wafer (e.g. BA2)    #
	# materials is a vector containing the names of each layer              #
	# d_vector is a vector containing the thickness of each layer [m]       #
	# doping_vector is a vector containing the doping of each layer [m^-3]  #
	# d_grading is the thickness of the grading between layers [m]          #
	# index_A_b is the index of the bottom A layer in the materials vector  #
	# index_A_t is the index of the top A layer in the materials vector     #
	# index_QW are the indices of the Quantum Wells in the materials vector #
	# index_peak is the index of the layer where the field should peak      #
	# etch_depth is the depth to etch to [m]                                #
	#                                                                       #
	# Outputs:                                                              #
	# .layer, .sol, .gain, .plt and .mac file are created in output_dir     #
	#                                                                       #
	# Crosslight Simulation Steps                                           #
	# 1. Generate PICS3D files using this code                              #
	# 2. Open the .sol file in PICS3D                                       #
	# 3. Click Generate Mesh, then Start, then Plot in the menu bar to      #
	#    generate 30 modes around the Bragg mode. Examine the output pdf    #
	#    to find Bragg mode, and identify refractive index using the        #
	#    output in the simulation tab of PICS3D                             #
	# 4. Adjust the .sol file to generate only the Bragg mode. This might   #
	#    involve some trial and error of adjusting the wave_boundary or     #
	#    index in direct_eigen                                              #
	# 5. Uncomment the scan lines at the bottom of the .sol file to run a   #
	#    complete scan                                                      #
	# 6. Uncomment the plot lines at the bottom of the .plt file to plot    #
	#    all results                                                        #
	# 7. If simulation does not match lab data well, start by fitting       #
	#    slope efficiency using loss (init wave backg_loss in .sol, or      #
	#    hole_carr_loss in .mac), then the threshold current using active   #
	#    region lines commented out in .sol                                 #
	#########################################################################
	PICS3D_dir = os.path.join(output_dir,"PICS3D",wafer_name)
	check_or_make_directory(PICS3D_dir)
	#Define geometry
	d_vector = [i*1e6 for i in d_vector]     #Convert from [m] to [µm]
	d_grading = d_grading*1e6                #Convert from [m] to [µm]
	etch_depth = etch_depth*1e6              #Convert from [m] to [µm]
	d_substrate = 150                        #Thickness of substrate [µm]
	ridge_width = 3                          #Full ridge width (will only simulate half) [µm]
	d_side_SiO2 = 0.2                        #Thickness of SiO2 on side of ridge [µm]
	d_SiO2 = 0.2                             #Thickness of SiO2 on top of etched region [µm]
	etched_width = 4                         #Width to simulate beside ridge [µm]
	wafer_name += "_2D"
	#Make output directory for PICS3D text results
	check_or_make_directory(os.path.join(PICS3D_dir,wafer_name+"_Output"))
	## Generate layer file
	layer_file = os.path.join(PICS3D_dir,wafer_name+'.layer')
	with open(layer_file, "w") as file:
		#Header
		file.write("$ "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("$ "+wafer_name+"\n")
		file.write("begin_layer\n")
		file.write("layer_input_convention grading_ref_point=yes\n")
		#Define Columns
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define columns                                                              $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("column column_num=1 w="+str(ridge_width/2)+" mesh_num=15 r=1\n")
		file.write("column_position label=left_edge location=left\n")
		file.write("column_position label=ridge_SiO2_col location=right\n")
		file.write("column column_num=2 w="+str(d_side_SiO2)+" mesh_num=6 r=1\n")
		file.write("column_position label=SiO2_air_col location=right\n")
		file.write("column column_num=3 w="+str(etched_width)+" mesh_num=6 r=1.1\n")
		file.write("column_position label=right_edge location=right\n")
		#Define bottom contacts
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define bottom contact                                                       $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("bottom_contact column_num=1 from=0 to="+str(ridge_width/2)+" contact_num=1 contact_type=ohmic\n")
		file.write("bottom_contact column_num=2 from=0 to="+str(d_side_SiO2)+" contact_num=1 contact_type=ohmic\n")
		file.write("bottom_contact column_num=3 from=0 to="+str(etched_width)+" contact_num=1 contact_type=ohmic\n")
		#Find initial depth
		#depth=t_cap+num_p_Bragg*(t_Bragg_1+t_Bragg_2)+t_ML_b+t_A_b+t_core_total+t_A_t+t_ML_t+num_n_Bragg*(t_Bragg_2+t_Bragg_1)
		depth = d_grading/2+sum(d_vector[1:-1])
		#Substrate
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define substrate                                                            $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Layer 0\n")
		file.write("layer_mater mater_lib=AlGaAs var_symbol1=x &&\n")
		file.write("column_num=1 var1=0 solve_wave=no n_doping="+str(-doping[0])+"\n")
		file.write("layer_mater mater_lib=AlGaAs var_symbol1=x &&\n")
		file.write("column_num=2 var1=0 solve_wave=no n_doping="+str(-doping[0])+"\n")
		file.write("layer_mater mater_lib=AlGaAs var_symbol1=x &&\n")
		file.write("column_num=3 var1=0 solve_wave=no n_doping="+str(-doping[0])+"\n")
		file.write("layer d="+str(d_substrate)+" n=45 r=-1.1\n")
		file.write("layer_position label=Substrate_top_interface location=top\n")
		if depth <= etch_depth:
			file.write("layer_position label=etch_interface location=top\n")
			file.write("layer_position label=SiO2_air_interface location=top\n")
		#Substrate to A layer
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define n-Bragg stack                                                        $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		for i in range(1,index_A_b):
			#grading
			layer_line=layer_line_graded(materials[i-1],materials[i],doping[i])
			add_layer(file,layer_line,d_grading,depth,etch_depth,d_SiO2,3,str(i-1)+" to "+str(i)+" grading")
			depth=depth-d_grading
			#layer
			layer_line=layer_line_regular(materials[i],doping[i])
			if i == index_A_b-1:
				mesh_points = 24 #give matching layer more mesh points
			elif i == index_A_b-2:
				mesh_points = 12 #give last Bragg layer more mesh points
			else:
				mesh_points = 6
			add_layer(file,layer_line,d_vector[i]-d_grading,depth,etch_depth,d_SiO2,mesh_points,i)
			depth=depth-d_vector[i]+d_grading
		# A layer (doped)
		if d_vector[index_A_b] > d_grading/2:
			#grading
			layer_line=layer_line_graded(materials[index_A_b-1],materials[index_A_b],doping[index_A_b])
			add_layer(file,layer_line,d_grading,depth,etch_depth,d_SiO2,3,str(index_A_b-1)+" to "+str(index_A_b)+" grading")
			depth=depth-d_grading
			#layer
			layer_line=layer_line_regular(materials[index_A_b],doping[index_A_b])
			add_layer(file,layer_line,d_vector[index_A_b]-d_grading/2,depth,etch_depth,d_SiO2,6,index_A_b)
		else:
			#grading
			layer_line=layer_line_graded(materials[index_A_b-1],materials[index_A_b],doping[index_A_b])
			add_layer(file,layer_line,d_grading/2,depth,etch_depth,d_SiO2,3,str(index_A_b-1)+" to "+str(index_A_b)+" grading")
			depth=depth-d_grading
			#layer
			layer_line=layer_line_regular(materials[index_A_b],doping[index_A_b])
			add_layer(file,layer_line,d_vector[index_A_b],depth,etch_depth,d_SiO2,6,index_A_b)
		depth=depth-d_vector[index_A_b]+d_grading/2
		# Core
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define core                                                                 $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		for i in range(index_A_b+1,index_A_t):
			#layer
			if i in index_QW:
				layer_line=layer_line_quantum_well(materials[i])
			else:
				layer_line=layer_line_regular(materials[i],doping[i])
			add_layer(file,layer_line,d_vector[i],depth,etch_depth,d_SiO2,6,i)
			depth=depth-d_vector[i]
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define p-Bragg stack                                                        $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		# A layer (doped)
		#layer
		layer_line=layer_line_regular(materials[index_A_t],doping[index_A_t])
		add_layer(file,layer_line,d_vector[index_A_t]-d_grading/2,depth,etch_depth,d_SiO2,6,index_A_t)
		depth=depth-d_vector[index_A_t]+d_grading/2
		#grading
		layer_line=layer_line_graded(materials[index_A_t],materials[index_A_t+1],doping[index_A_t])
		add_layer(file,layer_line,d_grading,depth,etch_depth,d_SiO2,3,str(index_A_t)+" to "+str(index_A_t+1)+" grading")
		depth=depth-d_grading
		#A layer to cap
		for i in range(index_A_t+1,len(materials)-2):
			#layer
			layer_line=layer_line_regular(materials[i],doping[i])
			if i == index_A_t+1:
				mesh_points = 24 #give matching layer more mesh points
			elif i == index_A_t+2:
				mesh_points = 12 #give first Bragg layer more mesh points
			else:
				mesh_points = 6
			add_layer(file,layer_line,d_vector[i]-d_grading,depth,etch_depth,d_SiO2,mesh_points,i)
			depth=depth-d_vector[i]+d_grading
			#grading
			layer_line=layer_line_graded(materials[i],materials[i+1],doping[i])
			add_layer(file,layer_line,d_grading,depth,etch_depth,d_SiO2,3,str(i)+" to "+str(i+1)+" grading")
			depth=depth-d_grading
		#Define cap
		layer_line="layer_mater mater_lib=AlGaAs var_symbol1=x column_num=1 var1=0 p_doping="+str(doping[-2])+"\n"
		add_layer(file,layer_line,d_vector[-2]-d_grading/2,depth,etch_depth,d_SiO2,3,'GaAs cap')
		file.write("layer_position label=Ridge_cap_interface location=bottom\n")
		file.write("layer_position label=Cap_contact_interface location=top\n")
		#Define top contact
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define top contact                                                          $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("top_contact column_num=1 from=0 to=1. contact_num=2 contact_type=ohmic\n")
		file.write("end_layer")
	##Generate plt file
	plt_file = os.path.join(PICS3D_dir,wafer_name+'.plt')
	with open(plt_file, "w") as file:
		#Header
		file.write("$ "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("$ "+wafer_name+"\n")
		file.write("begin_pstprc\n")
		file.write("plot_data plot_device=postscript\n")
		#Import Initial Data
		file.write("$ Import Data for initial plots\n")
		file.write("get_data  main_input="+wafer_name+".sol &&\n")
		file.write("sol_inf="+wafer_name+".out &&\n")
		file.write("xy_data=(1, 1) scan_data=(1, 1)\n")
		#Plot Refractive Index Profile
		file.write("$ Plot Refractive Index Profile\n")
		file.write("plot_1d variable=real_index x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		#Plot 2D Mode Profile
		file.write("$ Plot 2D Mode Profile\n")
		file.write("$ set value_to to the number of modes solved for (mode_num in .sol)\n")
		file.write("start_loop symbol=%j value_from=1 value_to=30 step=1\n")
		file.write("plot_2d variable=wave_intensity grid_sizes=(35, 35) mode_index=%j &&\n")
		file.write("xrange=(0,{:.7g})".format(d_side_SiO2+etched_width+ridge_width/2)+" yrange=({:.7g},".format(d_substrate)+"{:.7g})\n".format(d_substrate+d_grading/2+sum(d_vector[1:-1])))
		file.write("end_loop\n")
		#Plot 1D Mode Profile
		file.write("$ Plot 1D Mode Profile\n")
		file.write("$ set value_to to the number of modes solved for (mode_num in .sol)\n")
		file.write("start_loop symbol=%j value_from=1 value_to=30 step=1\n")
		file.write("plot_1d variable=wave_real_part mode_index=%j &&\n")
		file.write("from=(2E-5,{:.7g})".format(d_substrate)+" to=(2E-5,{:.7g})\n".format(d_substrate+d_grading/2+sum(d_vector[1:-1])))
		file.write("end_loop\n")
		file.write("$ Uncomment the rest once mode is found to plot additional parameters\n")
		#Plot Bandgap Profile
		file.write("$$ Plot 1D Mode Intensity (displays Gamma)\n")
		file.write("$plot_1d variable=wave_intensity mode_index=1 &&\n")
		file.write("$from=(2E-5,{:.7g})".format(d_substrate)+" to=(2E-5,{:.7g})\n".format(d_substrate+d_grading/2+sum(d_vector[1:-1])))
		file.write("$$ Plot bandgap\n")
		file.write("$plot_1d variable=bulk_bandgap x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_bandgap.txt\n")
		#Plot Equilibrium Potential
		file.write("$$ Plot equilibrium potential\n")
		file.write("$plot_1d variable=potential x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_potential_equilibrium.txt\n")
		#Import Remaining Data
		file.write("$$ Import Detailed Data - replace 14 with the number of .std files in the output\n")
		file.write("$$ - make change at bottom of .plt file too\n")
		file.write("$$ - only last file for geometry data\n")
		file.write("$$ - all files for scan data\n")
		file.write("$get_data main_input="+wafer_name+".sol &&\n")
		file.write("$sol_inf="+wafer_name+".out &&\n")
		file.write("$xy_data=(14, 14) scan_data=(1, 14)\n")
		#Plot Functions Of Current
		file.write("$$ Plot functions of current\n")
		file.write("$plot_1d variable=potential x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_potential.txt\n")
		file.write("$plot_scan scan_var=current_1 variable=voltage_1 scale_curr = 2 &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_voltage.txt\n")
		file.write("$plot_scan scan_var=laser_current_1 variable=all_mode_power &&\n")
		file.write("$facet=front scale_lit=2 scale_curr=2 &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_laser_power.txt\n")
		file.write("$plot_scan scan_var=current_1 variable=all_mode_power &&\n")
		file.write("$facet=front scale_lit=2 scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=temp_max scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=total_all_heat scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=total_joule_heat scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=total_optic_heat scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=total_recomb_heat scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=total_thomson_heat scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=efficiency yrange=(0, .3) scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=effective_index scale_curr=2\n")
		file.write("$plot_scan scan_var=current_1 variable=peak_gain scale_curr=2\n")
		#Plot Functions Of Geometry
		file.write("$$ Plot band diagram for one Bragg stack either side of core\n")
		file.write("$plot_1d variable=band from=[0.2E-04, {:.7g}]".format(d_substrate+d_grading/2+sum(d_vector[1:index_A_b-1]))+" to=[0.2E-04, {:.7g}]\n".format(d_substrate+d_grading/2+sum(d_vector[1:index_A_t+3])))
		file.write("$$ Plot functions of geometry\n")
		file.write("$plot_1d variable=lattice_temp x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=index_change x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=local_gain x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=elec_curr_y x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$yrange=(0, -5000) data_file="+wafer_name+"_Output/"+wafer_name+"_elec_curr.txt\n")
		file.write("$plot_1d variable=hole_curr_y x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$yrange=(0, -5000) data_file="+wafer_name+"_Output/"+wafer_name+"_hole_curr.txt\n")
		file.write("$plot_1d variable=recomb_rad x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_recomb_radiative.txt\n")
		file.write("$plot_1d variable=recomb_st x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_recomb_stimulated.txt\n")
		file.write("$plot_1d variable=recomb_srh x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_recomb_SRH.txt\n")
		file.write("$plot_1d variable=recomb_aug x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_recomb_Auger.txt\n")
		file.write("$plot_1d variable=rec_elec_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=elec_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_elec_concentration.txt\n")
		file.write("$plot_1d variable=linear_elec_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=rec_hole_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=hole_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_hole_concentration.txt\n")
		file.write("$plot_1d variable=linear_hole_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface\n")
		file.write("$plot_1d variable=elec_mobility x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_elec_mobility.txt\n")
		file.write("$plot_1d variable=hole_mobility x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_hole_mobility.txt\n")
		file.write("$plot_1d variable=all_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_all.txt\n")
		file.write("$plot_1d variable=joule_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_joule.txt\n")
		file.write("$plot_1d variable=optic_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_optical.txt\n")
		file.write("$plot_1d variable=peltier_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_peltier.txt\n")
		file.write("$plot_1d variable=recomb_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_recomb.txt\n")
		file.write("$plot_1d variable=thomson_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_thompson.txt\n")
		file.write("$plot_1d variable=radiation_heat x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_heat_radiation.txt\n")
		file.write("$plot_1d variable=absorption x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_absorption.txt\n")
		file.write("$gain_spectrum data_file="+wafer_name+"_Output/"+wafer_name+"_gain_biased.txt\n")
		file.write("$gain_spectrum variable=sp.rate &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_spont_rate_biased.txt\n")
		file.write("$gain_spectrum variable=rtg_spectrum &&\n")
		file.write("$data_file="+wafer_name+"_Output/"+wafer_name+"_power_spectrum.txt\n")
		file.write("$$$ Loop through current values to plot how some functions change with current\n")
		file.write("$$ - Loop through separately for each variable to order pdf by plot type\n")
		file.write("$$ - value_to must be changed in each loop to match the number of data files\n")
		file.write("$start_loop symbol=%i value_from=1 value_to=14 step=1\n")
		file.write("$get_data main_input="+wafer_name+".sol &&\n")
		file.write("$sol_inf="+wafer_name+".out &&\n")
		file.write("$xy_data=(%i, %i)\n")
		file.write("$plot_1d variable=band x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface $&&\n")
		file.write("$$data_file="+wafer_name+"_Output/"+wafer_name+"_band_bias_%i.txt\n")
		file.write("$end_loop\n")
		file.write("$start_loop symbol=%i value_from=1 value_to=14 step=1\n")
		file.write("$get_data main_input="+wafer_name+".sol &&\n")
		file.write("$sol_inf="+wafer_name+".out &&\n")
		file.write("$xy_data=(%i, %i)\n")
		file.write("$plot_1d variable=elec_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface $&&\n")
		file.write("$$data_file="+wafer_name+"_Output/"+wafer_name+"_elec_conc_%i.txt\n")
		file.write("$end_loop\n")
		file.write("$start_loop symbol=%i value_from=1 value_to=14 step=1\n")
		file.write("$get_data main_input="+wafer_name+".sol &&\n")
		file.write("$sol_inf="+wafer_name+".out &&\n")
		file.write("$xy_data=(%i, %i)\n")
		file.write("$plot_1d variable=hole_conc x_from_label=left_edge x_to_label=left_edge &&\n")
		file.write("$y_from_label=Substrate_top_interface y_to_label=Cap_contact_interface $&&\n")
		file.write("$$data_file="+wafer_name+"_Output/"+wafer_name+"_hole_conc_%i.txt\n")
		file.write("$end_loop\n")
		file.write("end_pstprc")
	##Generate sol file
	sol_file = os.path.join(PICS3D_dir,wafer_name+'.sol')
	pics3d_defaults = get_pics3d_defaults(wafer_name)
	with open(sol_file, "w") as file:
		#Header
		file.write("$ "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("$ "+wafer_name+"\n")
		file.write("begin\n")
		#Define Input Files
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define Input/Output Files                                                   $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("load_mesh mesh_inf="+wafer_name+".msh\n")
		file.write("include file="+wafer_name+".mater\n")
		file.write("include file="+wafer_name+".doping\n")
		file.write("use_macrofile macro1=algaas.mac\n")
		file.write("output sol_outf="+wafer_name+".out\n")
		#Define General Simulation Parameters
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define General Simulation Parameters                                        $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("more_output elec_mobility=yes hole_mobility=yes\n")
		file.write("$heat_flow fit_range=500 $comment out to ignore heating (not configured well)\n")
		#Define Recombination Interface
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define Recombination Interface                                              $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("interface model=recomb velocity_n=3e2 velocity_p=3e2 && $From Timmons_APL_1990\n")
		file.write("within_x1_label=ridge_SiO2_col within_x2_label=ridge_SiO2_col &&\n")
		file.write("within_y1_label=etch_interface within_y2_label=Cap_contact_interface\n")
		#Define Thermal Interfaces
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define Thermal Interfaces                                                   $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Heating is not yet configured well. Needs some work.\n")
		file.write("$ Heat sink at 300 K for bottom contact\n")
		file.write("$ Thermal conductance to air at 300 K for top contact\n")
		file.write("$ Contacts\n")
		file.write("contact num=1 thermal_type=1 lattice_temp=300 $bottom contact\n")
		file.write("contact num=2 thermal_type=3 extern_temp=300 thermal_cond=10 $top contact\n")
		file.write("$ Other Thermal Interfaces - thermal conductance to air at 300 K\n")
		file.write("thermal_interf thm_num=1 thm_type=3 thm_ext_temp=300 thm_cond=10 &&\n")
		file.write("within_x1_label=ridge_SiO2_col within_x2_label=SiO2_air_col &&\n")
		file.write("within_y1_label=SiO2_air_interface within_y2_label=Cap_contact_interface\n")
		file.write("thermal_interf thm_num=2 thm_type=3 thm_ext_temp=300 thm_cond=10 &&\n")
		file.write("within_x1_label=SiO2_air_col within_x2_label=right_edge &&\n")
		file.write("within_y1_label=etch_interface within_y2_label=SiO2_air_interface\n")
		#Define Active Region
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define Active Region Properties                                             $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("q_transport well_barrier_transport=trap_detrap_tau\n")
		file.write("$ Radiative Recomb Can be used as a fitting parameter for threshold current\n")
		file.write("$set_active_reg mater_label=QW analytical_recomb=yes $tau_scat=800.e-15\n")
		file.write("$radiative_recomb mater_label=QW value=1.8e16\n")
		#Define SiO2 Properties
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define SiO2 Properties                                                      $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("thermal_kappa value=1.3 mater_label=insulator\n")
		file.write("spec_heat value=680 mater_label=insulator\n")
		file.write("real_index value=1.4537 mater_label=insulator\n")
		#Simulate Bragg Mode
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Simulate Bragg Mode                                                         $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Adjust simluation boundaries so Arnoldi solver finds Bragg mode only\n")
		file.write("$if having trouble, try changing Cap_contact_interface to Ridge_cap_interface\n")
		file.write("$or ridge_SiO2_col to right_edge\n")
		file.write("optical_field profile=effective_index\n")
		file.write("wave_boundary x1_label="+pics3d_defaults[0]+" y1_label="+pics3d_defaults[1]+" &&\n")
		file.write("x2_label="+pics3d_defaults[2]+" y2_label="+pics3d_defaults[3]+"\n")
		file.write("init_wave fld_center=(0, {:.7g}) length=1000 backg_loss=500 &&\n".format(d_substrate+d_grading/2+sum(d_vector[1:index_peak])+d_vector[index_peak]/2))
		file.write("boundary_type=[2,1,1,1] init_wavel=0.78 wavel_range=[0.7, 0.83] mirror_ref=0.27\n")
		file.write("$ To find Bragg mode, set mode_num=30 and select_index to about 3.2\n")
		file.write("$ Once Bragg mode is found, set mode_num=1 and select_index just above mode\n")
		file.write("multimode mode_num="+pics3d_defaults[4]+"\n")
		file.write("direct_eigen select_modes=yes select_index="+pics3d_defaults[5]+"\n")
		#Define Scans
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Define Scans                                                                $\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ Note these currents are all current densities injected into the half model\n")
		file.write("$ Solve for thermal equilibrium (no bias)\n")
		file.write("newton_par damping_step=5. max_iter=100 print_flag=3\n")
		file.write("equilibrium\n")
		file.write("$ Uncomment the rest once mode is found to inject carriers\n")
		file.write("$$ Turn on voltage until current reaches 1 A/m\n")
		file.write("$newton_par damping_step=1.0 print_flag=3 res_tol=1e-4  &&\n")
		file.write("$var_tol=1e-4 opt_iter=15 stop_iter=10 max_iter=50 &&\n")
		file.write("$update_lateral_mode=no\n")
		file.write("$scan var=voltage_1 value_to=-2.0 &&\n")
		file.write("$init_step=1e-3 min_step=1.e-5 max_step=0.1 &&\n")
		file.write("$auto_finish=current_1 auto_until=1.0 auto_condition=above\n")
		file.write("$$ Scan current from 1 A/m to 5 A/m\n")
		file.write("$scan var=current_1 value_to=5 init_step=0.1 min_step=1e-6 max_step=1\n")
		file.write("$$ Change from simulating quasi-Fermi levels to carrier densities\n")
		file.write("$newton_par damping_step=1.0 print_flag=3 res_tol=1e-4  &&\n")
		file.write("$var_tol=1e-4 opt_iter=15 stop_iter=10 max_iter=50 &&\n")
		file.write("$change_variable=yes update_lateral_mode=no recover_prev_mqw=yes\n")
		file.write("$$ Scan current from 5 A/m up\n")
		file.write("$scan var=current_1 value_to=10 init_step=1e-3 min_step=1e-6 max_step=0.5\n")
		file.write("$scan var=current_1 value_to=30 init_step=1e-3 min_step=1e-6 max_step=0.5\n")
		file.write("$scan var=current_1 value_to=50 init_step=1e-3 min_step=1e-6 max_step=0.5\n")
		file.write("$scan var=current_1 value_to=70 init_step=1e-3 min_step=1e-6 max_step=0.1\n")
		file.write("$$ Adjust tolerance for future scans\n")
		file.write("$$ If solver fails to converge, change res_tol and var_tol to 5e-2\n")
		file.write("$newton_par damping_step=1.0 print_flag=3 res_tol=1e-4  &&\n")
		file.write("$var_tol=1e-2 opt_iter=15 stop_iter=10 max_iter=50 &&\n")
		file.write("$change_variable=yes update_lateral_mode=no recover_prev_mqw=yes\n")
		file.write("$start_loop symbol=%i value_from=90 value_to=210 step=20\n")
		file.write("$scan var=current_1 value_to=%i init_step=1e-3 min_step=1e-6 max_step=0.1\n")
		file.write("$end_loop\n")
		file.write("end")
	##Generate gain file
	gain_file = os.path.join(PICS3D_dir,wafer_name+'.gain')
	with open(gain_file, "w") as file:
		#Header
		file.write("$ "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("$ "+wafer_name+"\n")
		file.write("$ Right click > Gain Preview to run\n")
		#Define plots
		file.write("begin_gain\n")
		file.write("plot_data plot_device=postscript\n")
		file.write("include file="+wafer_name+".mater\n")
		file.write("gain_wavel wavel_range=(0.7 0.83) &&\n")
		file.write("conc_range=(5.e23 5.e24)  curve_number=10 data_point=200 &&\n")
		file.write("data_file = "+wafer_name+"_Output/"+wafer_name+"_gain.txt\n")
		file.write("sp.rate_wavel wavel_range=(0.7 0.83) &&\n")
		file.write("conc_range=(5.e23 5.e24)  curve_number=10 data_point=200 &&\n")
		file.write("data_file = "+wafer_name+"_Output/"+wafer_name+"_spont_rate.txt\n")
		file.write("gain_density wavel_range=(0.7 0.83) &&\n")
		file.write("conc_range=(5.e23 5.e24) data_point=20\n")
		file.write("end_gain")
	##Generate macro file
	macro_file = os.path.join(PICS3D_dir,'algaas.mac')
	with open(macro_file, "w") as file:
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("$ macro algaas\n")
		file.write("$ for bulk Al(x)Ga(1-x)As\n")
		file.write("$ Lattice matched to GaAs\n")
		file.write("$ refractive index modified for Iu_2022 model\n")
		file.write("$ [free-style]\n")
		file.write("$\n")
		file.write("$ Modified July 5, 2024 by Trevor Stirling:\n")
		file.write("$   - Gehrsitz formula for refractive index\n")
		file.write("$   - Iu_2022 correction for refractive index\n")
		file.write("$   - Added free carrier absoprtion losses\n")
		file.write("$ Note: 3d8 means 3e8 with double precision\n")
		file.write("$\n")
		file.write("$ Suggestion: Do not modify anything except hole_carr_loss which can be used\n")
		file.write("$             as a fitting parameter to match the slope efficiency\n")
		file.write("$\n")
		file.write("$ Typical use:\n")
		file.write("$   load_macro name=algaas var1=#x mater=#m var_symbol1=x\n")
		file.write("$ parameter_range x=[0 1]\n")
		file.write("$ parameter_range temper=[77 600]\n")
		file.write("$ parameter_range total_doping=[1.e20 1.e26]\n")
		file.write("$ parameter_range doping_n=[1.e20 1.e26]\n")
		file.write("$ parameter_range doping_p=[1.e20 1.e26]\n")
		file.write("$ parameter_range trap_1=[1.e18 1.e24]\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("begin_macro algaas\n")
		file.write("material type=semicond band_valleys=(1 1) &&\n")
		file.write("el_vel_model=n.gaas hole_vel_model=beta\n")
		file.write("\n")
		file.write("$$ Modifiable\n")
		file.write("elec_carr_loss value=1e-22 $typical value\n")
		file.write("hole_carr_loss value=150e-22 $used as a fitting parameter, 20-200 e-22\n")
		file.write("$$$$$$$$$$$$$$$$$$$$$$$$\n")
		file.write("\n")
		file.write("dielectric_constant variation=function\n")
		file.write("function(x)\n")
		file.write("13.1 - 3 * x\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("electron_mass variation=function\n")
		file.write("function(x)\n")
		file.write("for 0.<x<0.45\n")
		file.write("0.067 + 0.083*x\n")
		file.write("for 0.45<x<1.\n")
		file.write("0.85 - 0.14*x\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("hole_mass variation=function\n")
		file.write("function(x)\n")
		file.write("((0.087+0.063*x)**(3/2)+(0.62+0.14*x)**(3/2))**(2/3)\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("band_gap variation=function\n")
		file.write("function(x,temper)\n")
		file.write("for 0.<x<0.45\n")
		file.write("shift=-5.5e-4*temper**2/(temper+225)+9.4285712E-02;\n")
		file.write("1.424+1.247*x+shift\n")
		file.write("for 0.45<x<1.\n")
		file.write("shift=-5.5e-4*temper**2/(temper+225)+9.4285712E-02;\n")
		file.write("1.9+0.125*x+0.143*x*x+shift\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("affinity variation=function\n")
		file.write("function(x,temper)\n")
		file.write("for 0<x<0.45\n")
		file.write("offset=0.6;\n")
		file.write("shift=-5.5e-4*temper**2/(temper+225)+9.4285712E-02;\n")
		file.write("4.07-0.748*x-offset*shift\n")
		file.write("for 0.45<x<1.\n")
		file.write("offset=0.6;\n")
		file.write("shift=-5.5e-4*temper**2/(temper+225)+9.4285712E-02;\n")
		file.write("3.7964-0.14*x-offset*shift\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("electron_mobility variation=function\n")
		file.write("function(x,temper,doping_n,doping_p,trap_1)\n")
		file.write("for 0<x<0.45\n")
		file.write("fac=(300/temper)**2.3;\n")
		file.write("mu_max=0.85*exp(-18.516*x**2)*fac;\n")
		file.write("mu_min=0;\n")
		file.write("ref_dens=1.69d23;\n")
		file.write("alpha=0.436;\n")
		file.write("total_doping=doping_n+doping_p+trap_1;\n")
		file.write("mu_min+(mu_max-mu_min)/(1+(total_doping/ref_dens)**alpha)\n")
		file.write("for 0.45<x<1.\n")
		file.write("fac=(300/temper)**2.3;\n")
		file.write("mu_max=0.02*fac;\n")
		file.write("mu_min=0;\n")
		file.write("ref_dens=1.69d23;\n")
		file.write("alpha=0.436;\n")
		file.write("total_doping=doping_n+doping_p+trap_1;\n")
		file.write("mu_min+(mu_max-mu_min)/(1+(total_doping/ref_dens)**alpha)\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("hole_mobility variation=function\n")
		file.write("function(x,temper,doping_n,doping_p,trap_1)\n")
		file.write("dope=1.e22;\n")
		file.write("tfac1=(temper/300)**2.3;\n")
		file.write("tfac2=(temper/300)**1.5;\n")
		file.write("fac=1/(tfac1+1.6e-24*dope*tfac2);\n")
		file.write("mu_max=(0.04-0.048*x+0.02*x*x)*fac;\n")
		file.write("mu_min=0;\n")
		file.write("ref_dens=2.75d23;\n")
		file.write("alpha=0.395;\n")
		file.write("total_doping=doping_n+doping_p+trap_1;\n")
		file.write("mu_min+(mu_max-mu_min)/(1+(total_doping/ref_dens)**alpha)\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("beta_n value=2.\n")
		file.write("electron_sat_vel variation=function\n")
		file.write("function(x,temper)\n")
		file.write("for 0<x<0.45\n")
		file.write("fac=(300/temper)**2.3;\n")
		file.write("0.77e5*(1-0.44*x)*fac\n")
		file.write("for 0.45<x<1.\n")
		file.write("fac=(300/temper)**2.3;\n")
		file.write("8.e4*fac\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("beta_p value=1.\n")
		file.write("hole_sat_vel variation=function\n")
		file.write("function(temper)\n")
		file.write("dope=1.e22;\n")
		file.write("tfac1=(temper/300)**2.3;\n")
		file.write("tfac2=(temper/300)**1.5;\n")
		file.write("fac=1/(tfac1+1.6e-24*dope*tfac2);\n")
		file.write("1.d5*fac\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("norm_field value=4.e5\n")
		file.write("tau_energy value=1.e-13\n")
		file.write("radiative_recomb value=1.d-16\n")
		file.write("auger_n value=1.5e-42\n")
		file.write("auger_p value=1.5e-42\n")
		file.write("lifetime_n value=1.e-7\n")
		file.write("lifetime_p value=1.e-7\n")
		file.write("\n")
		file.write("real_index variation=function\n")
		file.write("function(x,temper,wavelength)\n")
		file.write("n=intern_func1(x,temper,wavelength);\n")
		file.write("corr=intern_func2(x,temper,wavelength);\n")
		file.write("n+corr\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("$ intern_func1: refractive index according to Gehrsitz\n")
		file.write("$ Journal of Applied Physics 87, 7825 (2000)\n")
		file.write("$ https://doi.org/10.1063/1.373462\n")
		file.write("intern_func1 variation=function\n")
		file.write("function(x,temper,wavelength)\n")
		file.write("$ wavelength should be in um\n")
		file.write("$ energies are quoted in 1/um in the paper\n")
		file.write("$ (a factor of 1.24 different from an eV)\n")
		file.write("$ Define constants\n")
		file.write("kB=8.617333262145e-5;\n")
		file.write("h=4.135667696e-15;\n")
		file.write("c=299792458;\n")
		file.write("$ Define fitting parameters\n")
		file.write("Ac0c0=5.9613;\n")
		file.write("Ac0c1=7.178e-4;\n")
		file.write("Ac0c2=-0.953e-6;\n")
		file.write("Ac1=-16.159;\n")
		file.write("Ac2=43.511;\n")
		file.write("Ac3=-71.317;\n")
		file.write("Ac4=57.535;\n")
		file.write("Ac5=-17.451;\n")
		file.write("C0c0=50.535;\n")
		file.write("C0c1=-150.7;\n")
		file.write("C0c2=-62.209;\n")
		file.write("C0c3=797.16;\n")
		file.write("C0c4=-1125;\n")
		file.write("C0c5=503.79;\n")
		file.write("C1c0=21.5647;\n")
		file.write("C1c1=113.74;\n")
		file.write("C1c2=-122.5;\n")
		file.write("C1c3=108.401;\n")
		file.write("C1c4=-47.318;\n")
		file.write("Eg0=1.5192;\n")
		file.write("S=1.8;\n")
		file.write("Edeb=15.9e-3;\n")
		file.write("St=1.1;\n")
		file.write("Et=33.6e-3;\n")
		file.write("E0c1=1.1308;\n")
		file.write("E0c2=0.1436;\n")
		file.write("E1c0c0=4.7171;\n")
		file.write("E1c0c1=-3.237e-4;\n")
		file.write("E1c0c2=-1.358e-6;\n")
		file.write("E1c1=11.006;\n")
		file.write("E1c2=-3.08;\n")
		file.write("C2=1.55e-3;\n")
		file.write("E2=sqrt(0.724e-3);\n")
		file.write("C3=2.61e-3;\n")
		file.write("E3=sqrt(1.331*1e-3);\n")
		file.write("$ Find energy in 1/um\n")
		file.write("E=1/wavelength;\n")
		file.write("$ Find Ac0: bottom right of Table II from paper\n")
		file.write("Ac0=Ac0c0+Ac0c1*temper+Ac0c2*temper**2;\n")
		file.write("$ Find A: (16) from paper\n")
		file.write("A=Ac0+Ac1*x+Ac2*x**2+Ac3*x**3+Ac4*x**4+Ac5*x**5;\n")
		file.write("$ Find C0: (16) from paper\n")
		file.write("C0=1/(C0c0+C0c1*x+C0c2*x**2+C0c3*x**3+C0c4*x**4+C0c5*x**5);\n")
		file.write("$ Find C0: (16) from paper\n")
		file.write("C1=C1c0+C1c1*x+C1c2*x**2+C1c3*x**3+C1c4*x**4;\n")
		file.write("$ Find E0c0: (11) from paper\n")
		file.write("E0c0=(Eg0+S*Edeb*(1-1/tanh(Edeb/2/kB/temper)) &&\n")
		file.write("+St*Et*(1-1/tanh(Et/2/kB/temper)));\n")
		file.write("E0c0=E0c0/(h*c*1e6);\n")
		file.write("$ Find E0: (16) from paper\n")
		file.write("E0=E0c0+E0c1*x+E0c2*x**2;\n")
		file.write("$ Find E1c0: bottom right of Table II from paper\n")
		file.write("E1c0=E1c0c0+E1c0c1*temper+E1c0c2*temper**2;\n")
		file.write("$ Find E1: (16) from paper\n")
		file.write("E1=sqrt(E1c0+E1c1*x+E1c2*x**2);\n")
		file.write("$ Find Reststrahl correction: (13) from paper\n")
		file.write("R=(1-x)*C2/(E2**2-E**2)+x*C3/(E3**2-E**2);\n")
		file.write("$ Final output: (12) from paper\n")
		file.write("nsqrd=A+C0/(E0**2-E**2)+C1/(E1**2-E**2)+R;\n")
		file.write("if (nsqrd>0)\n")
		file.write("sqrt(nsqrd)\n")
		file.write("else\n")
		file.write("0\n")
		file.write("endif\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("$ intern_func2: refractive index correction according to Iu_2022\n")
		file.write("intern_func2 variation=function\n")
		file.write("function(x,temper,wavelength)\n")
		file.write("$ wavelength should be in um\n")
		file.write("c = 3e8;\n")
		file.write("if(0.05<x<0.05)\n")
		file.write("correction=(0.3243*exp(0.6814*c/wavelength/1e8) &&\n")
		file.write("+1.609e-13*exp(7.609*c/wavelength/1e8))/1000*50-0.144119;\n")
		file.write("else if(0.28<x<0.28)\n")
		file.write("correction=0;\n")
		file.write("else if(0.337<x<0.337)\n")
		file.write("correction=0;\n")
		file.write("else if(0.<x<0.45)\n")
		file.write("correction=(4.704*exp(0.1839*(c/wavelength/1e8-3.738)/0.152) &&\n")
		file.write("+0.0321*exp(3.079*(c/wavelength/1e8-3.738)/0.152))/1000*9;\n")
		file.write("else if(0.45<x<1.)\n")
		file.write("correction=(1.72*exp(0.03609*(c/wavelength/1e8-3.747)/0.1473) &&\n")
		file.write("+0.2316*exp(0.3428*(c/wavelength/1e8-3.747)/0.1473))/1000*9;\n")
		file.write("endif\n")
		file.write("correction\n")
		file.write("end_function\n")
		file.write("\n")
		file.write("end_macro algaas")
	print("Created PICS3D Files in "+PICS3D_dir)

def Lumerical_file_generation(wafer_name, materials, d_vector, models="None", layer_names="Layer", d_n=0):
	#########################################################################
	# Creates files to run Lumerical Simulations                            #
	#                                                                       #
	# Inputs:                                                               #
	# wafer_name is a string containing the name of the wafer (e.g. BA2)    #
	# materials is a vector containing the names of each layer              #
	# d_vector is a vector containing the names of each layer [m]           #
	#                                                                       #
	# Outputs:                                                              #
	# .txt files are created in output_dir for each unique material as well #
	# as a script to generate the waveguide structure in Lumerical  and a   #
	# script to generate a DFB waveguide structure in Lumerical             #
	#                                                                       #
	# Lumerical Simulation Steps                                            #
	# 1. Copy existing simulation file to keep same meshes and materials    #
	# 2. Add material files for any new materials in Lumerical under        #
	#    Materials > Add > Sampled 3D Data > Import data...                 #
	# 3. Right click on waveguide at the left and copy the waveguide_file   #
	#    into the script field.                                             #
	# 4. Adjust the meshes to be centered on the structure if need be       #
	# 5. Run the simulation to find the Bragg mode                          #
	#########################################################################
	if models == "None":
		models = [models]*len(materials)
	if layer_names == "Layer":
		layer_names = [layer_names]*len(materials)
	if d_n == 0:
		d_n = [d_n]*len(materials)
	Lumerical_dir = os.path.join(output_dir,"Lumerical",wafer_name)
	check_or_make_directory(Lumerical_dir)
	#Define geometry
	d_substrate=10e-6     #Thickness of substrate [m]
	ridge_width=3e-6      #Full ridge width [m]
	dfb_narrow_width=1e-6 #Narrow ridge width of DFB [m]
	wl = [i*1e-9 for i in range(350,1600+1)]
	## Generate waveguide file
	waveguide_file = os.path.join(Lumerical_dir,wafer_name+'_2D_waveguide.txt')
	with open(waveguide_file, "w") as file:
		#Header
		file.write("# "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("# "+wafer_name+"\n")
		file.write("deleteall;\n")
		file.write("# Define Constants\n")
		file.write("#L=1e-6;\n")
		file.write("#w_ridge={:.6g};\n".format(ridge_width))
		file.write("#w_total=50e-6;\n")
		file.write("# Initialize variables\n")
		file.write("y=0;\n")
		file.write("#etch_depth=2e-6;\n")
		#Define Substrate
		file.write("# Substrate\n")
		file.write("addrect;\n")
		file.write("set(\"name\",\"substrate\");\n")
		file.write("set(\"material\",\""+materials[0]+"\");\n")
		file.write("set(\"z\",0);\n")
		file.write("set(\"z span\",L);\n")
		file.write("set(\"x\",0);\n")
		file.write("set(\"x span\",w_total);\n")
		file.write("set(\"y min\",y);\n")
		file.write("y=y+{:.6g};\n".format(d_substrate))
		file.write("set(\"y max\",y);\n")
		#Define other layers (except for top air layer)
		for i in range(1,len(d_vector)-1):
			file.write("# Layer {:d}\n".format(i))
			file.write("addrect;\n")
			file.write("set(\"name\",\""+layer_names[i]+"\");\n")
			if d_n[i] != 0:
				file.write("set(\"material\",\""+materials[i]+"_d_n\");\n")
			else:
				file.write("set(\"material\",\""+materials[i]+"\");\n")
			file.write("set(\"z\",0);\n")
			file.write("set(\"z span\",L);\n")
			file.write("set(\"x\",0);\n")
			file.write("set(\"x span\",w_total);\n")
			file.write("set(\"y min\",y);\n")
			file.write("y=y+{:.6g};\n".format(d_vector[i]))
			file.write("set(\"y max\",y);\n")
		#Etch ridge
		file.write("# Etch\n")
		file.write("addrect;\n")
		file.write("set(\"name\",\"etch\");\n")
		file.write("set(\"material\",\"etch\");\n")
		file.write("set(\"z\",0);\n")
		file.write("set(\"z span\",L);\n")
		file.write("set(\"x min\",-w_total/2);\n")
		file.write("set(\"x max\",-w_ridge/2);\n")
		file.write("set(\"y min\",y-etch_depth);\n")
		file.write("set(\"y max\",y);\n")
		file.write("# Etch\n")
		file.write("addrect;\n")
		file.write("set(\"name\",\"etch\");\n")
		file.write("set(\"material\",\"etch\");\n")
		file.write("set(\"z\",0);\n")
		file.write("set(\"z span\",L);\n")
		file.write("set(\"x min\",w_ridge/2);\n")
		file.write("set(\"x max\",w_total/2);\n")
		file.write("set(\"y min\",y-etch_depth);\n")
		file.write("set(\"y max\",y);")
	## Generate DFB file
	dfb_file = os.path.join(Lumerical_dir,wafer_name+'_2D_dfb.txt')
	with open(dfb_file, "w") as file:
		#Header
		file.write("# "+datetime.today().strftime('%Y_%m_%d')+"\n")
		file.write("# "+wafer_name+"\n")
		file.write("deleteall;\n")
		file.write("# Define Constants\n")
		file.write("L=1e-6;\n")
		file.write("w_thick={:.6g};\n".format(ridge_width))
		file.write("w_thin={:.6g};\n".format(dfb_narrow_width))
		file.write("w_total=50e-6;\n")
		file.write("d_substrate={:.6g};\n".format(d_substrate))
		file.write("# Initialize variables\n")
		file.write("#etch_depth=2e-6;\n")
		file.write("#fraction_narrow=0.5;\n")
		file.write("y=0;\n")
		file.write("unetched=1;\n")
		#Define other layers (except for top air layer)
		mat_string = "materials={"
		names_string = "names={"
		d_string = "d=["
		for i in range(1,len(d_vector)-1):
			if d_n[i] != 0:
				mat_string += "\""+materials[i]+"_d_n\","
			else:
				mat_string += "\""+materials[i]+"\","
			names_string += "\""+layer_names[i]+"\","
			d_string += "{:.6g},".format(d_vector[i])
		mat_string = mat_string[:-1]+"};\n"
		names_string = names_string[:-1]+"};\n"
		d_string = d_string[:-1]+"];\n"
		file.write(d_string)
		file.write(mat_string)
		file.write(names_string)
		file.write("d_total = sum(d)+d_substrate;\n")
		file.write("lambda=780e-9;\n")
		file.write("sio2_index=1.4537;\n")
		#Define Substrate
		file.write("# Substrate\n")
		file.write("addrect;\n")
		file.write("set(\"name\",\"substrate\");\n")
		file.write("set(\"material\",\""+materials[0]+"\");\n")
		file.write("set(\"z\",0);\n")
		file.write("set(\"z span\",L);\n")
		file.write("set(\"y min\",y);\n")
		file.write("y=y+d_substrate;\n")
		file.write("set(\"y max\",y);\n")
		file.write("set(\"x\",0);\n")
		file.write("set(\"x span\",w_total);\n")
		#Define other layers
		file.write("# All Other Layers\n")
		file.write("for(i=1:length(d)) {\n")
		file.write("	if (unetched == 1) {\n")
		file.write("		# While in unetched section, make full width layers\n")
		file.write("		addrect;\n")
		file.write("		set(\"name\",names{i});\n")
		file.write("		set(\"material\",materials{i});\n")
		file.write("		set(\"z\",0);\n")
		file.write("		set(\"z span\",L);\n")
		file.write("		set(\"x\",0);\n")
		file.write("		set(\"x span\",w_total);\n")
		file.write("		set(\"y min\",y);\n")
		file.write("		if (y+d(i) < d_total-etch_depth) {\n")
		file.write("			y=y+d(i);\n")
		file.write("			set(\"y max\",y);\n")
		file.write("		} else {\n")
		file.write("			d_etched = y+d(i)+etch_depth-d_total;\n")
		file.write("			y=d_total-etch_depth;\n")
		file.write("			set(\"y max\",y);\n")
		file.write("			unetched = 0;\n")
		file.write("			# Add partially etched section of layer if > 1 nm\n")
		file.write("			if (d_etched > 1e-9) {\n")
		file.write("				addrect;\n")
		file.write("				set(\"name\",names{i});\n")
		file.write("				set(\"material\",materials{i});\n")
		file.write("				set(\"z\",0);\n")
		file.write("				set(\"z span\",L);\n")
		file.write("				set(\"x\",0);\n")
		file.write("				set(\"x span\",w_thin);\n")
		file.write("				set(\"y min\",y);\n")
		file.write("				set(\"y max\",y+d_etched);\n")
		file.write("				# DFB averaged section - left\n")
		file.write("				addrect;\n")
		file.write("				set(\"name\",names{i});\n")
		file.write("				set(\"index\", getindex(materials{i}, c/lambda)*(1-fraction_narrow)+sio2_index* fraction_narrow);\n")
		file.write("				set(\"z\",0);\n")
		file.write("				set(\"z span\",L);\n")
		file.write("				set(\"x min\",-w_thick/2);\n")
		file.write("				set(\"x max\",-w_thin/2);\n")
		file.write("				set(\"y min\",y);\n")
		file.write("				set(\"y max\",y+d_etched);\n")
		file.write("				# DFB averaged section - right\n")
		file.write("				addrect;\n")
		file.write("				set(\"name\",names{i});\n")
		file.write("				set(\"index\", getindex(materials{i}, c/lambda)*(1-fraction_narrow)+sio2_index* fraction_narrow);\n")
		file.write("				set(\"z\",0);\n")
		file.write("				set(\"z span\",L);\n")
		file.write("				set(\"x min\",w_thin/2);\n")
		file.write("				set(\"x max\",w_thick/2);\n")
		file.write("				set(\"y min\",y);\n")
		file.write("				set(\"y max\",y+d_etched);\n")
		file.write("				y=y+d_etched;\n")
		file.write("			}\n")
		file.write("		}\n")
		file.write("	} else {\n")
		file.write("		# Unetched section\n")
		file.write("		addrect;\n")
		file.write("		set(\"name\",names{i});\n")
		file.write("		set(\"material\",materials{i});\n")
		file.write("		set(\"z\",0);\n")
		file.write("		set(\"z span\",L);\n")
		file.write("		set(\"x\",0);\n")
		file.write("		set(\"x span\",w_thin);\n")
		file.write("		set(\"y min\",y);\n")
		file.write("		set(\"y max\",y+d(i));\n")
		file.write("		# DFB averaged section - left\n")
		file.write("		addrect;\n")
		file.write("		set(\"name\",names{i});\n")
		file.write("		set(\"index\", getindex(materials{i}, c/lambda)*(1-fraction_narrow)+sio2_index* fraction_narrow);\n")
		file.write("		set(\"z\",0);\n")
		file.write("		set(\"z span\",L);\n")
		file.write("		set(\"x min\",-w_thick/2);\n")
		file.write("		set(\"x max\",-w_thin/2);\n")
		file.write("		set(\"y min\",y);\n")
		file.write("		set(\"y max\",y+d(i));\n")
		file.write("		# DFB averaged section - right\n")
		file.write("		addrect;\n")
		file.write("		set(\"name\",names{i});\n")
		file.write("		set(\"index\", getindex(materials{i}, c/lambda)*(1-fraction_narrow)+sio2_index* fraction_narrow);\n")
		file.write("		set(\"z\",0);\n")
		file.write("		set(\"z span\",L);\n")
		file.write("		set(\"x min\",w_thin/2);\n")
		file.write("		set(\"x max\",w_thick/2);\n")
		file.write("		set(\"y min\",y);\n")
		file.write("		set(\"y max\",y+d(i));\n")
		file.write("		y=y+d(i);\n")
		file.write("	}\n")
		file.write("}")
	## Generate materials files (except air)
	unique_index = np.unique(materials[:-1], return_index=True)[1]
	for i in unique_index:
		colour = [round(j*255) for j in colours(materials[i])]
		hex_colour = hex(colour[0])[2:]+hex(colour[1])[2:]+hex(colour[2])[2:]
		#Generate Material File
		material_file = os.path.join(Lumerical_dir,materials[i]+".txt")
		with open(material_file, "w") as file:
			#Header
			file.write("Wavelength [nm], n, k, Colour Hex: "+hex_colour)
			for j in range(len(wl)):
				n, k = Refractive_Index(wl[j],materials[i],models[i])
				file.write("\n{:.4g}, ".format(wl[j]*1e9)+"{:.6g}, ".format(n.real)+"{:.6g}".format(k.real))
		#If d_n != 0, Generate Second Material File
		if d_n[i] != 0:
			material_file = os.path.join(Lumerical_dir,materials[i]+"_d_n.txt")
			with open(material_file, "w") as file:
				file.write("Wavelength [nm], n, k, Colour Hex: "+hex_colour)
				for j in range(len(wl)):
					n, k = Refractive_Index(wl[j],materials[i],models[i])
					if wl[j] < 900e-9: #only add d_n to wavelengths near the bandgap
						n += d_n[i]
					file.write("\n{:.4g}, ".format(wl[j]*1e9)+"{:.6g}, ".format(n.real)+"{:.6g}".format(k.real))
	print("Created Lumerical Files in "+Lumerical_dir)