import math
import tensorflow as tf
import numpy as np
from sparsity_method_types import SparsityMethodTypes
from myconstants import conv_methods
import enum
# input_tensor_name:
# conv2d_self_name:
# input_shape:  Ih, Iw, Ic, In
# kernal_shape: Kw, Kh, Sw, Sh 
# output_shape: Ow, Oh,
#  
# 

class Conv_Layer(object):

  """A simple class for handling conv selfs."""
  def __init__(self, input_tensor_name, output_tensor_name, K, Kh, Kw, Sh, Sw, Oh, Ow, Ih, Iw, Ic, In=1, padding="VALID"):
    self.input_tensor_name                          =  input_tensor_name
    self.output_tensor_name                         =  output_tensor_name
    self.Kw                                         =  int(Kw)
    self.Kh                                         =  int(Kh)
    self.K                                          =  int(K)
    self.Sw                                         =  int(Sw)
    self.Sh                                         =  int(Sh)
    self.Ow                                         =  int(Ow)
    self.Oh                                         =  int(Oh)
    self.Ih                                         =  int(Ih)
    self.Iw                                         =  int(Iw)
    self.Ic                                         =  int(Ic)
    self.In                                         =  int(In)
    self.padding                                    =  padding
    # Save All the Compression Ratioes per Image in an single array
    self.CPO_cmpRatio                               =  np.empty(0, float)
    self.CPS_cmpRatio                               =  np.empty(0, float)
    self.MEC_cmpRatio                               =  np.empty(0, float)
    self.CSCC_cmpRatio                              =  np.empty(0, float)
    # Save All the densities Bounds
    self.density_bound_mec                          =  0
    # for different densities thats why we need a vector
    self.density_bound_cscc                         =  np.empty(0, float) 
    # Save All the densities per Image in an single array
    self.ru_batch                                   =  np.empty(0, float)
    self.lowering_den_batch                         =  np.empty(0, float)
    #CPS 
    self.pattern_width                              =  4 # CONST for our approach 
    self.patterns                                   =  np.empty(0, int)
    self.patterns_sum                               =  0

    self.lowering_density_channel                   = np.empty(0, float)
    self.feature_density_channel                    = np.empty(0, float)
  def __str__(self):
      try:
        s = ('\n=-=-=-=Conv_self=-=-=-= \n \
                  input_tensor_name: %s, output_tensor_name: %s \nIn: %d, Ic: %d, Ih: %d, Iw: %d \
                \nKh: %d, Kw: %d, K: %d, padding: %s \
                \nSh: %d, Sw: %d \
                \nOh: %d, Ow: %d \
                \nFeature Map shape rows: %d , cols: %d, channels: %d \
                \nAfter padding Shape rows: %d , cols: %d, channels: %d \
                \nLowering nnz = %d ,feature map nnz = %d \
                \nDensity : Feature Map--> [ %f <-> %f ] <--Lowering Matrix' %
                (
                self.input_tensor_name, self.output_tensor_name, \
                self.In, self.Ic, self.Ih, self.Iw, \
                self.Kh, self.Kw, self.K, self.padding, self.Sh, self.Sw, self.Oh, self.Ow, \
                self.Iw_padded, self.Ih_padded, self.Ic, \
                self.Iw_padded, self.Ih_padded, self.Ic, \
                self.tot_nz_lowering, self.tot_nz_feature, \
                self.ru, self.lowering_density
                )
            )
      except:
                s = ('=-=-=-=Conv_self=-=-=-= \ninput_tensor_name: %s, output_tensor_name: %s \nIn: %d, Ic: %d, Ih: %d, Iw: %d \
                \nKh: %d, Kw: %d, K: %d, padding: %s \
                \nSh: %d, Sw: %d \
                \nOh: %d, Ow: %d \
                \nMISSING INFO after Calcuations are CALLED\\DONE\n' % (self.input_tensor_name, self.output_tensor_name, \
                self.In, self.Ic, self.Ih, self.Iw, \
                self.Kh, self.Kw, self.K, self.padding, self.Sh, self.Sw, self.Oh, self.Ow))
      return s

  def padding_cal(self):
      #Padding Feature Map
      if (self.padding =='SAME'):
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
          # print(self)
          print(("Calculated --> Ow = %d, Oh = %d, Actual --> Ow = %d, Oh = %d")%(cal_Ow, cal_Oh, self.Ow, self.Oh))
          print("ERROR in padding in dimensions")
          # exit(0)

      paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
      paddings = tf.convert_to_tensor(paddings, dtype=tf.int32)
      self.paddings = paddings.eval(session=tf.compat.v1.Session())
       
  # Here we should add the padding, lowering matrix reps, etc
  
  def image_padding(self, feature_maps):
    #   One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    feature_maps        = tf.pad(feature_maps, self.paddings, "CONSTANT",constant_values=0)
    feature_maps        = feature_maps.eval(session=tf.compat.v1.Session())
    self.Ih_padded      = feature_maps.shape[0]
    self.Iw_padded      = feature_maps.shape[1]
    self.tot_nz_feature = np.size(feature_maps[feature_maps != 0.0])
    resol_feature       = self.Iw_padded*self.Ih_padded*self.Ic
    self.ru             = self.tot_nz_feature/resol_feature
    self.ru_batch       = np.append(self.ru_batch,self.ru)
    

    return feature_maps
  
  def lowering_rep(self, feature_maps):
    lowering_matrix                    = np.empty((self.Ow,0), int)
    self.last_channel_cal              = self.Ic
    #############################################
    # Not the most important information we need
    self.lower_desity_count     = 0
    self.feature_desity_count   = 0
    self.both_feature_lowering  = 0
    #############################################
    
    for idx in range(self.last_channel_cal):
        m_f = feature_maps[:, :, idx]  
        # Here we creates the lowering Matrix for MEC and CSCC
        sub_tmp = np.empty((0,self.Kw*self.Ih_padded), int)
        if ((self.Kw>1) or (self.Kh>1)):
            for col_int in range(0, self.Iw_padded, self.Sw):
                if (col_int+self.Kw)>self.Iw_padded :
                    break

                x       = m_f[:,col_int:col_int+self.Kw] # col_int+kw-1 without 1 bec it the stop element
                x       = x.ravel(order='K') # K -> for row major && F -> for col major
                x       = np.reshape(x,(1,np.size(x)))
                sub_tmp = np.append(sub_tmp, x, axis=0)
        else:
            sub_tmp     = m_f.transpose();
        
        lowering_matrix = np.append(lowering_matrix, sub_tmp, axis=1)
        self.lowering_density_channel = np.append( self.lowering_density_channel, np.size(sub_tmp[sub_tmp != 0.0])/(sub_tmp.shape[0]*sub_tmp.shape[1]))
        self.feature_density_channel = np.append(self.feature_density_channel, np.size(m_f[m_f != 0.0])/(m_f.shape[0]*m_f.shape[1]))

        if (self.feature_density_channel[idx] < self.lowering_density_channel[idx] ):
            # print (("Density per channel : Feature Map--> [ %f < %f ] <--Lowering Matrix ")%(feature_desity_channel[idx], lowering_density_channel[idx]))
            self.lower_desity_count = self.lower_desity_count + 1
        elif (self.feature_density_channel[idx] > self.lowering_density_channel[idx] ):
            # print (("Density per channel : Feature Map--> [ %f > %f ] <--Lowering Matrix ")%(feature_desity_channel[idx], lowering_density_channel[idx]))
            self.feature_desity_count = self.feature_desity_count + 1
        else:
            self.both_feature_lowering = self.both_feature_lowering + 1
            # print (("Density per channel : Feature Map--> [ %f = %f ] <--Lowering Matrix ")%(feature_desity_channel[idx], lowering_density_channel[idx]))
    return lowering_matrix

  def cal_density(self, lowering_matrix):  
    self.lowering_shape = np.shape(lowering_matrix)
    resol_lowering = self.lowering_shape[0]*self.lowering_shape[1]
    resol_lowering_ = self.Ow*self.Ih*self.Kw*self.Ic
    # CHECK
    if (resol_lowering != resol_lowering_):
      print("ERROR in the shape of the in Lowering Matrix")
    self.tot_nz_lowering = np.size(lowering_matrix[lowering_matrix != 0.0])
    
    # Here if you want to select a specific end for your calc 
    if (self.last_channel_cal != self.Ic):
      self.tot_nz_feature = np.size(feature_maps[feature_maps[:,:,::last_channel] != 0.0])
      resol_feature = self.Iw_padded*self.Ih_padded*last_channel
      self.ru = self.tot_nz_feature/resol_feature
      self.ru_batch = np.append(self.ru_batch, self.ru)

    self.lowering_density = self.tot_nz_lowering/resol_lowering
    self.lowering_den_batch = np.append(self.lowering_den_batch,self.lowering_density)

  def print_all(self):
    # print("\n############ Print Layer object ############\n")
    print(("Feature Map shape rows: %d , cols: %d, channels: %d ")
          %(self.Iw_padded, self.Ih_padded, self.Ic))
    print(("After padding with --> %s Method || Shape rows: %d , cols: %d, channels: %d ")
          %(self.padding, self.Iw_padded, self.Ih_padded, self.Ic))
    print(("lowering matrix shape rows: %d , cols: %d")
          %(self.lowering_shape[0], self.lowering_shape[1]))
    print("Lowering nnz = %d ,feature map nnz = %d"
          %(self.tot_nz_lowering, self.tot_nz_feature))
    print(("Density : Feature Map--> [ %f <-> %f ] <--Lowering Matrix ")
          %(self.ru, self.lowering_density))
    print("Density Winner Counts : Feature Map --> [%d, (BOTH)-> %d , %d] <-- Lowering Matrix"
          %(self.feature_desity_count, self.both_feature_lowering , self.lower_desity_count))
    # print("\n############ END ***\/** object ############\n")
    
