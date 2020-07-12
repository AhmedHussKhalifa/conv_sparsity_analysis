# This file should contain all equations for calculating the space

# Force floating point division
from __future__ import division
import math

# Import conv layer
from conv_layer import Conv_Layer

# Calculates the required memory units for the **CPO** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCPO(is_for_density_calc=True):

    if is_for_density_calc:
        space = 0
    else:
        space = 1
    return space

# Calculates the required memory units for the **MEC**  method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceMEC(is_for_density_calc=True):

    if is_for_density_calc:
        space = 0
    else:
        space = 1
    return space

# Calculates the required memory units for the **CSCC** method
# is_for_density is a flag to know whether we should use the assumptions for space calculations or not
def getSpaceCSCC(is_for_density_calc=True):
    
    if is_for_density_calc:
        space = 0
    else:
        space = 1
    return space

# Calculates the required memory units for the **Im2Col** method
def getSpaceIm2Col():
    space = 0
    return space


