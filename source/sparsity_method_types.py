import enum


# Usage: print(conv_methods['CPO'])
class SparsityMethodTypes(enum.Enum):   
    # This class desribes the possible method types for convolution (CPO, CPS, MEC, CSCC, Im2Col)
    CPO              = 1
    CPS              = 2
    MEC              = 3
    CSCC             = 4
    Im2Col           = 5

    def getModelByValue(val):
        return SparsityMethodTypes(val).name

    
    def getModelByName(val):
        return SparsityMethodTypes[val].value