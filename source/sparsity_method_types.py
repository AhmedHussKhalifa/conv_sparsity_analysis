import enum


# Usage: print(conv_methods['CPO'])
class SparsityMethodTypes(enum.Enum):   
    # This class desribes the possible method types for convolution (CPO, CPS, MEC, CSCC, Im2Col)
    CPO              = 1
    CPS              = 2
    MEC              = 3
    CSCC             = 4
    SparseTensor     = 5
    Im2Col           = 6

    Org_Density      = 10
    Bound_MEC        = 20
    Bound_CSCC       = 30

    def getModelByValue(val):
        return SparsityMethodTypes(val).name

    
    def getModelByName(val):
        return SparsityMethodTypes[val].value
