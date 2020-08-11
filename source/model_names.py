import enum



#models             = {  'IV1':         1,
#                        'IV3':         2,
#                        'IV4':         3,
#                        'MobileNet':   4,
#                        'MobileNetV2': 5,
#                        'ResNet-V2-101': 6,
#                        'ResNet-V2-50':  7,
#                        'Pnasnet_Large': 8,
#                        'nasnet_mobile': 9,
#                        'EfficientNet' : 10,
#                        'InceptionResnetV2': 11,
#                        'Vgg16'        : 12,
#                        'Vgg19'        : 13
#
# }

# This class desribes the possible model names for our inference code.
class Models(enum.Enum):

    IV1         = 1
    IV3         = 2
    IV4         = 3
    mobileNet   = 4
    mobileNetV2 = 5
    resnet101   = 6
    resnet50    = 7
    Pnasnet_Large = 8
    nasnet_mobile = 9
    EfficientNet = 10
    inceptionresnetv2 = 11
    Vgg16        = 12
    Vgg19        = 13
    
    

    def getModelByValue(val):
        return Models(val).name

    def getModelByName(val):
        return Models[val].value
