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
        space = (layer.Ow*layer.Kw)/layer.sw + (layer.Kw/layer.sw) + layer.Ow + 1 + 2*(layer.ru*layer.Ih*layer.Iw)
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
        space = 0
    else:
        space = 1
    return space

# Calculates the required memory units for the **CSCC** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCSCC(layer, is_for_density_calc=True):
    
    if is_for_density_calc:
        space = 0
    else:
        space = 1
    return space

# Calculates the required memory units for the **Im2Col** method
def getSpaceIm2Col(layer, is_for_density_calc=True):
    if is_for_density_calc:
        space = 0
    else:
        space = (layer.Kw/layer.Sw)*(layer.Ow+1)+2*layer.Ih*layer.Iw*layer.ru
    return space



