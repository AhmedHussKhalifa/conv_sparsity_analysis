import enum

# This class desribes the possible model names for our inference code.
# Usage: print(conv_methods['CPO'])
class Sparsity_Method_Types(enum.Enum):   
	# This class desribes the possible method types for convolution (CPO, CPS, MEC, CSCC, Im2Col)
	conv_methods         = {'CPO'   :      1,
	                        'CPS'   :      2,
	                        'MEC'   :      3,
	                        'CSCC'  :      4,
	                        'Im2Col':      5
	}
    def getModelByValue(val):
        return conv_methods(val).name

    def getModelByName(val):
        return conv_methods[val].value