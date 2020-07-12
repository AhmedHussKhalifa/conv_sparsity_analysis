import enum


# This class desribes the possible model names for our inference code.
class Models(enum.Enum):

    IV1         = 1
    IV3         = 2
    IV4         = 3
    mobileNet   = 4
    mobileNetV2 = 5
    resnet101   = 6
    Pnasnet_Large = 7
    nasnet_mobile = 8
    EfficientNet = 9
    inceptionresnetv2 = 10
    Vgg16        = 11
    Vgg19        = 12
    
    

    def getModelByValue(val):
        return Models(val).name

    def getModelByName(val):
        return Models[val].value
