# This file should contain all equations for calculating the space

# Force floating point division
from __future__ import division
import math

# Import conv layer
from conv_layer import Conv_Layer
from myconstants import conv_methods
import numpy as np

# Calculates the required memory units for the **CPO** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCPO(layer, is_for_density_calc=True):
    # Used to caluclate the Compression Ratio
    # we should multiply by Ic here, create seperate functions for this
    if (layer.Kw%layer.Sw) == 0:
        space = layer.In*(layer.Kw/layer.Sw)*(layer.Ow+1)+2*(layer.Ih*layer.Iw*layer.ru*layer.Ic) 
    elif (layer.Kw%layer.Sw) != 0:
        space = math.ceil(layer.Kw/layer.Sw)*(layer.Ow+1)+2*(layer.Ih*layer.Iw*layer.ru*layer.Ic)
    return space

def getSpaceCPS(layer, is_for_density_calc=True):
    # use the assumption of Ic = 1 && In = 1
    if (layer.Kw%layer.Sw) == 0:
        space = layer.In*(layer.Kw/layer.Sw)*(layer.Ow+1) 
                + (layer.Ih*layer.Iw*layer.Ic*sum(layer.ru_batch)) 
                + (layer.patterns_sum)
    elif (layer.Kw%layer.Sw) != 0:
        space = layer.In*math.ceil(layer.Kw/layer.Sw)*(layer.Ow+1) 
                + (layer.Ih*layer.Iw*layer.Ic*sum(layer.ru_batch)) 
                + (layer.patterns_sum)
    return space

# Calculates the required memory units for the **MEC**  method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceMEC(layer, is_for_density_calc=True):
    space = layer.Ow*layer.Kw*layer.Ih*layer.In*layer.Ic
    return space

# Calculates the required memory units for the **CSCC** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCSCC(layer, is_for_density_calc=True): 
    space = (layer.Ow + 1) + (2*layer.lowering_density*layer.Ow*layer.Ih*layer.Kw)
    return space

# Calculates the required memory units for the **Im2Col** method
def getSpaceIm2Col(layer, is_for_density_calc=True):
    space = layer.Ow*layer.Oh*layer.Ic*layer.Kw*layer.Kh
    return space

# Calculates the required density bound for MEC vs CPO
def getDensityBoundMEC(layer):
    density_bound_mec = ((layer.Ow*layer.Kw*layer.Ih) - ((layer.Ow*layer.Kw)/layer.Sw ) 
                            - (layer.Kw/layer.Sw) - layer.Ow - 1)
    density_bound_mec = density_bound_mec / (2*layer.Ih*layer.Iw)
    # without considering Ic != 1
    # density_bound_mec = density_bound_mec / (2*layer.Ih*layer.Iw*layer.Ic)
    return density_bound_mec

# Calculates the required density bound for CSCC vs CPO
def getDensityBoundCSCC(layer):
    density_bound_cscc = (layer.Kw/layer.Iw) * ( (layer.Ow*layer.lowering_density) 
                            - (layer.Ow+1)/(2*layer.Ih*layer.Sw) )
    # without considering Ic != 1
    # density_bound_cscc = (layer.Kw/layer.Iw) * ( (layer.Ow*layer.Ic*layer.lowering_density) 
    #                         - (layer.Ow+1)/(2*layer.Ih*layer.Sw) )
    return density_bound_cscc

# Calculates the required density bound for lowering matric 

def getDensityBoundLoweringDensityCSCC(layer):
    density_bound_lowering_density =  (layer.Ow+1) / (2*layer.Ih*layer.Sw*layer.Ow)
    if (density_bound_lowering_density>layer.lowering_density):
        print("WARNING: CSCC (Winner) Lowering denisty is lower than the bound")
    return density_bound_lowering_density

def getCR(layer, method_type, Im2col_space = 1, is_for_density_calc=False):
    if (method_type == conv_methods['CPO']):
        layer.CPO_cmpRatio = np.append(layer.CPO_cmpRatio, 
            getSpaceCPO(layer, is_for_density_calc)/Im2col_space)
    elif (method_type == conv_methods['CPS']):
        layer.CPS_cmpRatio = np.append(layer.CPS_cmpRatio, 
            getSpaceCPS(layer, is_for_density_calc)/Im2col_space)
    elif (method_type == conv_methods['MEC']):
        layer.MEC_cmpRatio = np.append(layer.MEC_cmpRatio, 
            getSpaceMEC(layer, is_for_density_calc)/Im2col_space)
    elif (method_type == conv_methods['CSCC']):
        layer.CSCC_cmpRatio = np.append(layer.CSCC_cmpRatio, 
            getSpaceCSCC(layer, is_for_density_calc)/Im2col_space)
    elif (method_type == conv_methods['Im2Col']):
        return getSpaceIm2Col(layer, is_for_density_calc)

def getDensityBound(layer, method_type):
    if (method_type == conv_methods['MEC']):
        layer.density_bound_mec = getDensityBoundMEC(layer)
    elif (method_type == conv_methods['CSCC']):
        layer.density_bound_cscc = np.append(layer.density_bound_cscc, getDensityBoundCSCC(layer))