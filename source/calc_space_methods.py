# This file should contain all equations for calculating the space

# Force floating point division
from __future__ import division
import math

# Import conv layer
from conv_layer import Conv_Layer

# Calculates the required memory units for the **CPO** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCPO(layer, is_for_density_calc=True):
    if is_for_density_calc:
        # use the assumption of Ic = 1 && In = 1
        space = (layer.Ow*layer.Kw)/layer.Sw + (layer.Kw/layer.Sw) + layer.Ow + 1 + 2*(layer.ru*layer.Ih*layer.Iw)
    else:
        # we should multiply by Ic here, create seperate functions for this
        if (layer.Kw%layer.Sw) == 0:
            space = (layer.Kw/layer.Sw)*(layer.Ow+1)+2*(layer.Ih*layer.Iw*layer.ru*layer.Ic) 
        elif (layer.Kw%layer.Sw) != 0:
            space = math.ceil(layer.Kw/layer.Sw)*(layer.Ow+1)+2*(layer.Ih*layer.Iw*layer.ru*layer.Ic)
    return space

def getSpaceCPS(layer, is_for_density_calc=True):
    if is_for_density_calc:
        # use the assumption of Ic = 1 && In = 1
        space = 1
    else:
        # we should multiply by Ic here, create seperate functions for this
        space = 0
    return space

# Calculates the required memory units for the **MEC**  method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceMEC(layer, is_for_density_calc=True):
    if is_for_density_calc:
        space = layer.Ow*layer.Kw*layer.Ih
    else:
        space = layer.Ow*layer.Kw*layer.Ih*layer.Ic
    return space

# Calculates the required memory units for the **CSCC** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCSCC(layer, is_for_density_calc=True): 
    if is_for_density_calc:
        space = (layer.Ow + 1) + (2*layer.lowering_density*layer.Ow*layer.Ih*layer.Kw)
    else:
        space = (layer.Ow + 1) + (2*layer.lowering_density*layer.Ow*layer.Ih*layer.Kw*layer.Ic)
    return space

# Calculates the required memory units for the **Im2Col** method
def getSpaceIm2Col(layer, is_for_density_calc=True):
    if is_for_density_calc:
        space = layer.Ow*layer.Oh*layer.Kw*layer.Kh
    else:
        space = layer.Ow*layer.Oh*layer.Ic*layer.Kw*layer.Kh
    return space

# Calculates the required density bound for MEC vs CPO
def getDensityBoundMEC(layer):
    density_bound_mec = ((layer.Ow*layer.Kw*layer.Ih) - ((layer.Ow*layer.Kw)/layer.Sw ) 
                            - (layer.Kw/layer.Sw) - layer.Ow - 1)
    density_bound_mec = density_bound_mec / (2*layer.Ih*layer.Iw)
    return density_bound_mec

# Calculates the required density bound for CSCC vs CPO
def getDensityBoundCSCC(layer):
    density_bound_cscc = (layer.Kw/layer.Iw) * ( (layer.Ow*layer.lowering_density) 
                            - (layer.Ow+1)/(2*layer.Ih*layer.Sw) )
    return density_bound_cscc

# Calculates the required density bound for lowering matric 

def getDensityBoundLoweringDensityCSCC(layer):
    density_bound_lowering_density =  (layer.Ow+1) / (2*layer.Ih*layer.Sw*layer.Ow)
    if (density_bound_lowering_density>layer.lowering_density):
        print("WARNING: CSCC (Winner) Lowering denisty is lower than the bound")
    return density_bound_lowering_density
