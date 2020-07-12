import enum


# This class desribes the possible code modes for our inference code.
class CodeMode(enum.Enum):

    Org              = 1
    All              = 2

    # val should be integer
    def getCodeName(val):
    	try:
    		return CodeMode(val).name
    	except:
    		return -1

   	# val should be string
    def getCodeIndex(val):
    	try:
    		return CodeMode[val].value
    	except:
    		return -1
