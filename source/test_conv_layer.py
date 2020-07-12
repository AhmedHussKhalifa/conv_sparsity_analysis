

from conv_layer import Conv_Layer


#__init__(self, input_tensor_name, output_tensor_name, Kw, Kh, Sw, Sh, Ow, Oh):

input_tensor_name  = 'input'
output_tensor_name = 'output'
Kw                 = 8
Kh                 = 8
Sw                 = 1
Sh                 = 1
Ow                 = 1
Oh                 = 1


current_layer = Conv_Layer(input_tensor_name, output_tensor_name, Kw, Kh, Sw, Sh, Ow, Oh)

#Conv_Layer c = (self, input_tensor_name, output_tensor_name, Kw, Kh, Sw, Sh, Ow, Oh)

print(current_layer.input_tensor_name)

