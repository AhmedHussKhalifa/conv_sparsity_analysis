# Force floating point division
from __future__ import division
import math

# this I is after padding
Ih  = 147
Iw  = 147
Kw  = 3
sw  = 1
ru = 0.574504

# This Ih, Iw after padding... there is a padding equation needs to be done
Ow  = 1 + (Iw - Kw)/sw

print('Ih, Iw, Kw, sw, Ow: ', Ih, Iw, Kw, sw, Ow)
print('-------------\n')


########## Calculate S1, S2 thru Ow

S1 = Ow*Kw/sw + Kw/sw + Ow + 1 + 2*ru*Ih*Iw
S2 = Ow*Kw*Ih


print('Value of ru, S1, S2, S2-S1', ru, S1, S2, S2-S1)
print('-------------\n')

#######

print('[Ow] Calculate the density bound thru Ow')

density_bound = (Ow*Kw*Ih - (Ow*Kw)/sw - Kw/sw - Ow - 1)
density_bound = density_bound / (2*Ih*Iw)

print('Density bound thru Ow: ', density_bound)

# Clip the ru
ru = min(1, max(0, density_bound))

print('\n')
print('[Ow] ru plugged into equation: ', ru)
print('\n-----\n')

S1 = Ow*Kw/sw + Kw/sw + Ow + 1 + 2*ru*Ih*Iw
S2 = Ow*Kw*Ih

print('S1: ', S1, ', S2: ', S2)
print('S2-S1: ', S2 - S1)
print('\n-----\n')

################

