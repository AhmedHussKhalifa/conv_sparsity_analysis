

# input_tensor_name:
# conv2d_layer_name:
# input_shape:  Ih, Iw, Ic, In
# kernal_shape: Kw, Kh, Sw, Sh 
# output_shape: Ow, Oh,
#  
# 

class Conv_Layer(object):
  """A simple class for handling conv layers."""

  def __init__(self, input_tensor_name, output_tensor_name, K, Kh, Kw, Sh, Sw, Oh, Ow, Ih, Iw, Ic, In=1, padding="VALID"):
    
    self.input_tensor_name  =  input_tensor_name
    self.output_tensor_name =  output_tensor_name
    self.Kw                 =  int(Kw)
    self.Kh                 =  int(Kh)
    self.K                  =  int(K)
    self.Sw                 =  int(Sw)
    self.Sh                 =  int(Sh)
    self.Ow                 =  int(Ow)
    self.Oh                 =  int(Oh)
    self.Ih                 =  int(Ih)
    self.Iw                 =  int(Iw)
    self.Ic                 =  int(Ic)
    self.In                 =  int(In)
    self.padding            =  padding

  def __str__(self):
      s = ('=-=-=-=Conv_Layer=-=-=-= \ninput_tensor_name: %s, output_tensor_name: %s \nIn: %d, Ic: %d, Ih: %d, Iw: %d \
              \nKh: %d, Kw: %d, K: %d, padding: %s \
              \nSh: %d, Sw: %d \
              \nOh: %d, Ow: %d' % (self.input_tensor_name, self.output_tensor_name, \
              self.In, self.Ic, self.Ih, self.Iw, \
              self.Kh, self.Kw, self.K, self.padding, self.Sh, self.Sw, self.Oh, self.Ow))
      return s

  # Here we should add the intermediate matrix reps, etc
