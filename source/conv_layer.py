import math
import tensorflow as tf
import numpy as np
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

  def padding_cal(self):
      #Padding Feature Map
      if (self.padding =='SAME'):
          # print("Padding --> ","SAME")
          cal_Ow = math.ceil(float(self.Ih) / float(self.Sh))
          cal_Oh  = math.ceil(float(self.Iw) / float(self.Sw))

          pad_along_height = max((self.Oh - 1) * self.Sh +
                              self.Kh - self.Ih, 0)
          pad_along_width = max((self.Ow - 1) * self.Sw +
                             self.Kw - self.Iw, 0)
          pad_top = pad_along_height // 2
          pad_bottom = pad_along_height - pad_top
          pad_left = pad_along_width // 2
          pad_right = pad_along_width - pad_left

          # print("tensorflow --> ",pad_top , pad_bottom, pad_left, pad_right)
      elif (self.padding =='VALID'):
          # print("Padding --> ","VALID")
          cal_Oh  = math.ceil(float(self.Ih - self.Kh + 1) / abs(float(self.Sh)))
          cal_Ow   = math.ceil(float(self.Iw - self.Kw + 1) / abs(float(self.Sw)))
          pad_top = 0
          pad_bottom = 0
          pad_left = 0
          pad_right = 0
      else:
          print("ERROR in padding at inputs")
          exit(0)

      if ((self.Ow != cal_Ow) or (self.Oh != cal_Oh)):
          print(self)
          print(("Calculated --> Ow = %d, Oh = %d, Actual --> Ow = %d, Oh = %d")%(cal_Ow, cal_Oh, self.Ow, self.Oh))
          print("ERROR in padding in dimensions")
          # exit(0)

      paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
      paddings = tf.convert_to_tensor(paddings, dtype=tf.int32)
      self.paddings = paddings.eval(session=tf.compat.v1.Session())
       
  def padding_image(self, feature_maps):
      #   One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
      feature_maps    = tf.pad(feature_maps, self.paddings, "CONSTANT",constant_values=0)
      feature_maps    = feature_maps.eval(session=tf.compat.v1.Session())
      self.Ih_padded  = feature_maps.shape[0]
      self.Iw_padded  = feature_maps.shape[1]
      self.tot_nz_feature = np.size(feature_maps[feature_maps != 0.0])
      resol_feature   = self.Iw_padded*self.Ih_padded*self.Ic
      self.ru = self.tot_nz_feature/resol_feature
      return feature_maps
  # Here we should add the intermediate matrix reps, etc
