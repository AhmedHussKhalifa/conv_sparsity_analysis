# Reference link: https://machinethink.net/blog/how-fast-is-my-model/

# -*- coding: utf-8 -*-
# Hossam Amer & Sepideh Shaterian
# Inception image recognition attempt v1
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np
import re
import os
import time




# Excel sheet stuff:
import xlrd
from xlwt import *
from xlutils.copy import copy

import copy
import cv2
import math 
import glob 
import time 
import imagenet
import sys
import argparse


from model_names import Models
from code_modes  import CodeMode
from conv_layer import Conv_Layer
from calc_space_density_methods import *
from sparsity_method_types import SparsityMethodTypes

# import imagenet_preprocessing as ipp
from preprocessing import inception_preprocessing, vgg_preprocessing
import pickle 

human_labels = imagenet.create_readable_names_for_imagenet_labels()
# np.set_printoptions(threshold=sys.maxsize, precision=3)
np.set_printoptions(threshold=sys.maxsize)

# Import all constants
from myconstants import *

# Import process
import multiprocessing
from multiprocessing import Process
import subprocess

def create_graph(): 
    with tf.gfile.FastGFile(MODEL_PATH + Frozen_Graph[FLAGS.model_name], 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


# Gets image data for different DNNs
def get_image_data(first_jpeg_image):
    if FLAGS.model_name  == 'InceptionResnetV2':
        image_data = tf.read_file(first_jpeg_image)
        image_data = tf.image.decode_jpeg(image_date, channels=3)
    elif FLAGS.model_name == 'IV3':
        image_data = tf.gfile.FastGFile(first_jpeg_image, 'rb').read()
    elif FLAGS.model_name == 'AlexNet':
        imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        img           = cv2.imread(first_jpeg_image)
        img           = cv2.resize(img.astype(np.float32), (resized_dimention[FLAGS.model_name], resized_dimention[FLAGS.model_name]))
        img          -= imagenet_mean
        image_data    = img.reshape((1, resized_dimention[FLAGS.model_name], resized_dimention[FLAGS.model_name], 3))

    else:
         image_data              = readImage(first_jpeg_image)

    return image_data


# Runs the DNN for analysis 
def run_DNN_for_analysis(sess, ic, c, input_tensor_list, first_input_tensor, image_data):
    if FLAGS.model_name  == 'InceptionResnetV2' or  FLAGS.model_name == 'IV3':
        current_feature_map, input_to_feature_map             = sess.run([c, sess.graph.get_tensor_by_name(input_tensor_list[ic])],
            {first_input_tensor[0]: image_data})
    elif FLAGS.model_name  == 'MobileNetV2':
        current_feature_map, input_to_feature_map             = sess.run([c, sess.graph.get_tensor_by_name(input_tensor_list[ic])],
            {"input:0": sess.run(image_data)})
    elif FLAGS.model_name == 'AlexNet':
        current_feature_map, input_to_feature_map             = sess.run([c, sess.graph.get_tensor_by_name(input_tensor_list[ic])],
            {"Placeholder:0": image_data, 'Placeholder_1:0': 1})
    else:
        current_feature_map, input_to_feature_map             = sess.run([c, sess.graph.get_tensor_by_name(input_tensor_list[ic])]
       , {first_input_tensor[0]: sess.run(image_data)})
    return current_feature_map, input_to_feature_map





def get_DNN_info_general(sess, first_jpeg_image, n_images = 1):

    graph_def = sess.graph.as_graph_def(add_shapes=True)
    total_maccs = 0

   

    all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
    all_layers  = []
    input_tensor_list = []
    conv_tensor_list  = []
    padding_tensor_list = []
    sw_tensor_list = []
    sh_tensor_list = []
    first_input_tensor = []
    k_tensor_list  = []
    kw_tensor_list = []
    kh_tensor_list = []
    num_classes    = 2 # binary in our problem

    depth_wise_conv_tensor_list = []

    # Total macs of things you trained
    trained_total_maccs = 0
    logits_total_maccs_binary  = 0
    logits_total_maccs_imagenet = 0

    # Trainable parameters:
    logits_trainable_params_binary = 0
    logits_trainable_params_imagenet = 0


    # loop on all nodes in the graph
    for  nid, n in enumerate(graph_def.node):
        try: 
            if nid == 0:
                first_input_tensor.append(n.name + ':0')
           
            if not nid % 150:
                print(nid)

            if ('softmax/logits' in n.name or 'FC/MatMul' in n.name) and 'MatMul' in n.op:
                #print(n)
                weights_tensor_name = n.input[1] + ':0'
                weights_tensor      = sess.graph.get_tensor_by_name(weights_tensor_name)
                # print(weights_tensor)

                logits_total_maccs_binary    = weights_tensor.shape[0] * num_classes
                logits_total_maccs_imagenet  = weights_tensor.shape[0] * weights_tensor.shape[1]

                # Calculate the total trainable parameters
                logits_trainable_params_binary += logits_total_maccs_binary + 1
                logits_trainable_params_binary += logits_total_maccs_imagenet + 1


            # Trainable total macs:
            if 'Logits' in n.name and 'Conv2D' in n.op:
                conv_tensor_params_name = n.input[1] + ':0'
                conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
                filter_shape            = conv_tensor_params.shape
                Cin                     = filter_shape[0]
                Kh                      = filter_shape[1]
                Kw                      = filter_shape[2]
 
                # Hossam: Cout is 1001 in this case when imageNet - In our case it is 2 classes (binary)
                # Cout                    = filter_shape[3]
                Cout = num_classes
                
                # Get image data
                image_data = get_image_data(first_jpeg_image)
                
                # Run the first image
                input_dw_tensor_list = [n.input[0] + ':0']
                output_tensor_name = n.name + ':0'
                c                  = [sess.graph.get_tensor_by_name(output_tensor_name)]
                current_feature_map, input_to_feature_map = run_DNN_for_analysis(sess, 0, c, input_dw_tensor_list, first_input_tensor, image_data)
                current_feature_map = current_feature_map[0]
                _, Oh, Ow, _   = current_feature_map.shape
                In, Ih, Iw, Ic = input_to_feature_map.shape
                
                logits_total_maccs_binary    = Kw * Kh * Cin * Oh * Ow * Cout
                logits_total_maccs_imagenet  = Kw * Kh * Cin * Oh * Ow * filter_shape[3]
            

                # Calculate the total trainable parameters
                logits_trainable_params_binary += Kw * Kh * Cout * In + (1 * 1 * Cout)
                logits_trainable_params_imagenet += Kw * Kh * filter_shape[3] * In + (1 * 1 * filter_shape[3])

        
            # Total macs:
            if 'DepthwiseConv2d' in n.op: 
                #if 'padding' in n.attr.keys():

                #print(n.op)
                # conv_tensor_params_name = n.name + ':0'
                # conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
                # print(conv_tensor_params)

                conv_tensor_params_name = n.input[1] + ':0'
                conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
                #print(conv_tensor_params)
                filter_shape            = conv_tensor_params.shape
                Kh                      = filter_shape[0]
                Kw                      = filter_shape[1]
                Cin                     = filter_shape[2]
                Cexp                    = Cin * filter_shape[3]

                # Get image data
                image_data = get_image_data(first_jpeg_image)
                
                # Run the first image
                input_dw_tensor_list = [n.input[0] + ':0']
                output_tensor_name = n.name + ':0'
                c                  = [sess.graph.get_tensor_by_name(output_tensor_name)]
                current_feature_map, input_to_feature_map = run_DNN_for_analysis(sess, 0, c, input_dw_tensor_list, first_input_tensor, image_data)
                current_feature_map = current_feature_map[0]
                _, Oh, Ow, _   = current_feature_map.shape
                In, Ih, Iw, Ic = input_to_feature_map.shape
                
                total_maccs    += Kh * Kw * Cexp* Oh * Ow
            
            # elif 'project/Conv2D' in n.name and n.op == 'Conv2D':
            #     # print(n)
            #     # print(n.op)

            #     conv_tensor_params_name = n.input[1] + ':0'
            #     conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
            #     print(conv_tensor_params)
            #     filter_shape            = conv_tensor_params.shape
            #     Cin                     = filter_shape[0]
            #     Kh                      = filter_shape[1]
            #     Kw                      = filter_shape[2]
            #     Cout                    = filter_shape[3]
            #     Cexp                    = Cin * Cout

            #     # Get image data
            #     image_data = get_image_data(first_jpeg_image)
                
            #     # Run the first image
            #     input_dw_tensor_list = [n.input[0] + ':0']
            #     output_tensor_name = n.name + ':0'
            #     c                  = [sess.graph.get_tensor_by_name(output_tensor_name)]
            #     current_feature_map, input_to_feature_map = run_DNN_for_analysis(sess, 0, c, input_dw_tensor_list, first_input_tensor, image_data)
            #     current_feature_map = current_feature_map[0]
            #     _, Oh, Ow, _   = current_feature_map.shape
            #     In, Ih, Iw, Ic = input_to_feature_map.shape
                
            #     total_maccs    += Cexp * Oh * Ow * Cout
            #     #exit(0)


            elif ('Conv2D' in n.name or 'convolution' in n.name) and '_bn_' not in n.name and 'Logits' not in n.name and 'logits' not in n.name:

                output_tensor_name = n.name + ':0'
                input_tensor_name = n.input[0] + ':0'
                input_tensor      = sess.graph.get_tensor_by_name(input_tensor_name)
                input_tensor_list.append(input_tensor_name)
                conv_tensor_list.append(sess.graph.get_tensor_by_name(output_tensor_name))



                if 'padding' in n.attr.keys():
                    padding_type = n.attr['padding'].s.decode(encoding='utf-8')
                    padding_tensor_list.append(padding_type)
                if 'strides' in n.attr.keys():
                    art_tensor_name = n.name + ':0'
                    strides_list = [int(a) for a in n.attr['strides'].list.i]
                    Sh           = strides_list[1]
                    Sw           = strides_list[2]
                    sh_tensor_list.append(Sh)
                    sw_tensor_list.append(Sw)
                
                conv_tensor_params_name = n.input[1] + ':0'
                conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
                filter_shape            = conv_tensor_params.shape
                Kh                      = filter_shape[0]
                Kw                      = filter_shape[1]
                K                       = filter_shape[3]
                k_tensor_list.append(K)
                kh_tensor_list.append(Kh)
                kw_tensor_list.append(Kw)



        except ValueError:
            print('%s is an Op.' % n.name)
   
    
    # ensure that created lists have the same length
    assert(len(sw_tensor_list) == len(sh_tensor_list) == len(padding_tensor_list)
            == len(conv_tensor_list) == len(input_tensor_list))
    assert(len(k_tensor_list) == len(kh_tensor_list) == len(kw_tensor_list) == len(sw_tensor_list))
    
    # Write the header in text file
    print('Fetched Primary information about DNN...')


    # Loop through convolutions to get the conv dimensions
    for ic, c in enumerate(conv_tensor_list):
        
        # Get image data
        image_data = get_image_data(first_jpeg_image)
        
        # Run the first image
        current_feature_map, input_to_feature_map = run_DNN_for_analysis(sess, ic, c, input_tensor_list, first_input_tensor, image_data)
        
        _, Oh, Ow, _   = current_feature_map.shape
        In, Ih, Iw, Ic = input_to_feature_map.shape
        K, Kh, Kw,     = k_tensor_list[ic], kh_tensor_list[ic], kw_tensor_list[ic]
        Sh, Sw         = sh_tensor_list[ic], sw_tensor_list[ic]
        padding_type   = padding_tensor_list[ic]

        input_tensor_name  = input_tensor_list[ic]
        output_tensor_name = c.name

         
        
        # Create the conv_layer
        conv_layer = Conv_Layer(input_tensor_name, output_tensor_name, K, Kh, Kw, Sh, Sw, Oh, Ow, Ih, Iw, Ic, In, padding=padding_type)
        conv_layer.padding_cal()
        # print(input_tensor_name, output_tensor_name)
        # Calculate the densities
        input_to_feature_map             = np.squeeze(input_to_feature_map)
    
        # lowering_matrix                 = conv_layer.preprocessing_layer(np.squeeze(input_to_feature_map))
        lowering_matrix                 = conv_layer.preprocessing_layer(input_to_feature_map)

        #print('********* PADDING TYPE ', conv_layer.padding, ' PADDINGS: ' , conv_layer.paddings, ' Input: ', input_tensor_name, ' out: ', output_tensor_name,
        #        ' Ih: ', conv_layer.Ih)
        
        # Get all layers
        all_layers.append(conv_layer)

        print('[%s] Analyzed Conv Node %d' % (FLAGS.model_name, ic))

    
    print('Reset the session')

    print('Extracted info for these %d layers... No Risk No Fun :) ' % len(all_layers))
    print('This code should work without exit.... Done!')


    # print_tensors_list_simple(sess)

    # total_flops = tf.profiler.profile(sess.graph,\
    #  options=tf.profiler.ProfileOptionBuilder.float_operation())

    # # total_trainable_params = tf.profiler.profile(sess.graph,\
    # #  options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())


    all_conv_trainable_params = 0
    for layer in all_layers:
        # Normal Conv: K × K × Cin × Hout × Wout × Cout
        # if layer.Sw == 2:
        #     print(layer.Ih, layer.Iw, layer.Oh, layer.Ow, layer.padding)
        #     exit(0)
        total_maccs += layer.Kw * layer.Kh * layer.Ic * layer.Oh * layer.Ow * layer.K

        # all conv trainable params:
        all_conv_trainable_params += layer.Kw * layer.Kh * layer.K * layer.In + (1 * 1 * layer.K)

        # if 'expand' in layer.output_tensor_name:
        #     print(layer.output_tensor_name)
        #     print(total_maccs)

        #     #Cexp = (Cin × expansion_factor)
        #     #expansion_layer = Cin × Hin × Win × Cexp
        #     print(layer.Ic * 1.4 * layer.Ic * layer.Ih * layer.Iw)
        #     print(layer.Sh, layer.Sw)
        #     exit(0)

    # One layer before for trainable Conv2D
    bf_last_imageNet_layer_trained_total_maccs = 0
    bf_last_selector_layer_trained_total_maccs = 0

    # One layer before trainable parameters:
    bf_last_trainable_params = 0

    if FLAGS.model_name == 'MobileNetV2':
        layer                   = all_layers[-1]

        # Calcualte the selector bf last layer:
        bf_last_selector_layer_trained_total_maccs = layer.Kw * layer.Kh * layer.Ic * layer.Oh * layer.Ow * num_classes

        # Calculate the last layer trained total macs assuming 1000 classes (to be subtracted)
        bf_last_imageNet_layer_trained_total_maccs = layer.Kw * layer.Kh * layer.Ic * layer.Oh * layer.Ow * layer.K

        # Calculate the total trainable params:
        bf_last_trainable_params  =  layer.Kw * layer.Kh * layer.K * layer.In + (1 * 1 * layer.K)

    elif FLAGS.model_name == 'IV3':
        for layer in all_layers:
            # if 'mixed_10' not in layer.output_tensor_name and 'cell_11' not in layer.output_tensor_name:
            if 'mixed_10' not in layer.output_tensor_name:
                continue

            print(layer.output_tensor_name, 'mixed_10')
            # Calcualte the selector bf last layer:
            bf_last_selector_layer_trained_total_maccs += layer.Kw * layer.Kh * layer.Ic * layer.Oh * layer.Ow * layer.K

            print('progress:', bf_last_selector_layer_trained_total_maccs)

            # Calcualte the selector bf last layer:
            bf_last_imageNet_layer_trained_total_maccs += layer.Kw * layer.Kh * layer.Ic * layer.Oh * layer.Ow * layer.K


            # Calculate the total trainable params:
            bf_last_trainable_params  +=  layer.Kw * layer.Kh * layer.K * layer.In + (1 * 1 * layer.K)



    # Get anything before trainable macs: (logits are already excluded)
    anything_before_selector_maccs = total_maccs -  bf_last_imageNet_layer_trained_total_maccs

    # Get anything before trainable parameters:
    anything_before_selector_trainable_params = all_conv_trainable_params - bf_last_trainable_params

    # Adjust your total macs for the model assuming ImageNet
    total_maccs        = anything_before_selector_maccs + bf_last_imageNet_layer_trained_total_maccs + logits_total_maccs_imagenet

    
    # Adjust your total for the model assuming binary choice
    total_maccs_binary = anything_before_selector_maccs + bf_last_selector_layer_trained_total_maccs + logits_total_maccs_binary

    # Calculate the trainable params
    total_trainable_params_imagenet = anything_before_selector_trainable_params + bf_last_trainable_params + logits_trainable_params_imagenet


    # Calculate the trainable params binary:
    total_trainable_params_binary = bf_last_trainable_params + logits_trainable_params_binary

    # Calculate selector maccs
    selector_maccs = total_maccs_binary + 8*(bf_last_selector_layer_trained_total_maccs + logits_total_maccs_binary)


    # Calculate difference in flops for the selector
    maccs_delta = (selector_maccs - total_maccs_binary)

    #print(bf_last_imageNet_layer_trained_total_maccs)

    print('ImageNet Stuff')
    print('MACs for %s: %d' % (FLAGS.model_name, total_maccs))

    print('Binary Stuff')
    print('MACs for %s binary: %d' % (FLAGS.model_name, total_maccs_binary))
    print('MACs for %s Selector: %d' % (FLAGS.model_name, selector_maccs))
    print('MACs increase between Selector and default binary %s: %d' % (FLAGS.model_name, maccs_delta))
    print('anything_before_selector_maccs: %d ;  trained_section: %d' % (anything_before_selector_maccs, bf_last_selector_layer_trained_total_maccs 
        + logits_total_maccs_binary))
    print('Trainable FC: %d  --- Trainable ModuleBefore: %d ' % (logits_trainable_params_binary, bf_last_selector_layer_trained_total_maccs))

    # print('Total trainable parameters imagenet for %s is: %d' % (FLAGS.model_name, total_trainable_params_imagenet))
    # print('Selector Residual trainable parameters binary for %s is: %d' % (FLAGS.model_name, total_trainable_params_binary))

# if total_flops_per_layer / 1e9 > 1:   # for Giga Flops
#     print(total_flops_per_layer/ 1e9 ,'{}'.format('GFlops'))
# else:
#     print(total_flops_per_layer / 1e6 ,'{}'.format('MFlops'))

    # print(total_flops)
    # #print(total_trainable_params)

    exit(0)
    return all_layers



# This function does not work for IV1 because some of the output shapes 
# are not written in the forzen grph


def print_tensors_list(sess):
    tensor_names = [n.name for n in sess.graph.as_graph_def().node]

    for tensor in tensor_names:
        itensor   = tensor + ":0"
        ac_tensor = sess.graph.get_tensor_by_name(itensor)
        print(tensor, ac_tensor.shape)

def print_tensors_list_simple(sess):
    tensor_names = [n.name for n in sess.graph.as_graph_def().node]

    for tensor in tensor_names:
       print(tensor)


def print_tensors(sess):
    
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    print('Law Yoom ye3ady')
    for  nid, n in enumerate(graph_def.node):
        try:
            current_tensor = sess.graph.get_tensor_by_name(n.name + ':0')
            print('--Tensor----')
            print(nid)
            print(graph_def.node[0])
            exit(0)
            #print(current_tensor)
            if 'conv2d_params' in current_tensor.name:
                print('--====---- CONVO--======----')
                print(current_tensor)
           
            if 'strides' in n.attr.keys():
                print (n.name, ' Strides: ' , [int(a) for a in n.attr['strides'].list.i])
            if 'Conv2D' in current_tensor.name:
                print('**************')
                #exit(0)
        except ValueError:
            print('%s is an Op.' % n.name)
        if '_output_shapes' in n.attr.keys():
            try:
                shapes = n.attr['_output_shapes']
                print(n.name, [int(a.size) for a in shapes.list.shape[0].dim])
            except:
                continue


def  print_ops(sess):
  constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
  for constant_op in constant_ops:
    print(constant_op.name)  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide the warning information from Tensorflow - annoying...



def construct_qf_list():

    qp_list = []

    for i in range(10, 110, 10):
        qp_list.append(i)
    return qp_list


"""Makes sure the folder exists on disk.
Args:
  dir_name: Path string to the folder we want to create.
"""
def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def readImage(current_jpeg_image):
    
    img_size                = resized_dimention[FLAGS.model_name]
    image_data              = tf.read_file(current_jpeg_image)
    image_data              = tf.image.decode_jpeg(image_data, channels=3)

    # (InceptionResnetV2 does not need prepreocessing)
    if models[FLAGS.model_name] != Models.inceptionresnetv2.value:
        if models[FLAGS.model_name] < Models.inceptionresnetv2.value:
            image_data              = inception_preprocessing.preprocess_image(image_data, img_size, img_size, is_training=False)
        else:
            # Different preprocessing for vgg19 and vgg16 (this reading is faulty now)

            image_data = vgg_preprocessing.preprocess_image(image_data, img_size, img_size, is_training=False)
            # image_data              = inception_preprocessing.preprocess_image(image_data, img_size, img_size, is_training=False)
            

    image_data              = tf.expand_dims(image_data, 0)

    # print(image_data)
    # exit()

    return image_data




def calc_FLops():

    qf_list  = construct_qf_list()
    img_size   = resized_dimention[FLAGS.model_name]
    t        = 0
    top1_acc = 0 
    top5_acc = 0 

    for imgID in range(FLAGS.START, FLAGS.END):

        startTime = time.time()

        original_img_ID = imgID
        actual_idx = original_img_ID
       
        if (actual_idx - 1) % 50 == 0 or actual_idx == FLAGS.START:
            
            config = tf.ConfigProto(device_count = {'GPU': 0})
            sess = tf.Session(config=config)
            create_graph()
            softmax_tensor = sess.graph.get_tensor_by_name(final_tensor_names[FLAGS.model_name])
            print('New session group has been created')

        # Get the DNN info for all layers at the start
        if actual_idx == FLAGS.START:

            first_jpeg_image      = org_image_dir + '/shard-' + str(0) + '/' +  str(1) + '/' + 'ILSVRC2012_val_' + str(1).zfill(8) + '.JPEG'
            #first_jpeg_image      = org_image_dir + '/shard-' + str(0) + '/' +  str(1) + '/' + 'ILSVRC2012_val_' + str(10).zfill(8) + '.JPEG'
            all_layers_info       = get_DNN_info_general(sess, first_jpeg_image)
            exit(0)
            #all_layers_info = get_DNN_info(sess)

        
    return top1_acc , top5_acc 
              

def ensure_select_flag(x):
    return x == CodeMode.getCodeName(1) or x == CodeMode.getCodeName(2) or x == CodeMode.getCodeName(3)

def ensure_model_name(x):
    try:
        x = models[x]
        return True
    except:
        return False

############################################################
# Note this code does nort work for IV4 since its output tensor only takes batch size 1 
# python3-tf run_inference.py --select Org --model_name IV3 --END 2
def main(_):

    model_path      =  WORKSPACE_DIR + FLAGS.model_name
    ensure_dir_exists(model_path)

    if not ensure_select_flag(FLAGS.select):
        print('\n\n[Error] Your input select flag should either be \'%s\' or \'%s\' ' 
            % (CodeMode.getCodeName(1), CodeMode.getCodeName(2)))
        exit(0)

    if not ensure_model_name(FLAGS.model_name):
        print('\n\n[Error] Your input model name is wrong or not supported.' )
        print('Here are the names of the supported models:')
        for key in models:
            print('\t\'%s\'' % key)
        exit(0)

    if models[FLAGS.model_name] == Models.inceptionresnetv2.value:
        if FLAGS.batch_size != 1:
            print('\n\n [Error] InceptionResnetV2 does not support batching in inference.')
            exit(0)


    start = time.time()
    num_images =  50000
    

    mac, flops = calc_FLops()

    # Hossam: Disable the batching for inference for now - Later we need to add it
    #if models[FLAGS.model_name] == Models.inceptionresnetv2.value or models[FLAGS.model_name] == Models.IV3.value:
    #    top1_count, top5_count = readAndPredictOptimizedImageByImage()
    #else:
    #    top1_count, top5_count = readAndPredictLoopOptimized()
    #top1_accuracy = top1_count / num_images *100 
    #top5_accuracy = top5_count / num_images *100 
    #print('top1_accuracy == ', top1_accuracy ,'top5_accuracy == ',  top5_accuracy )

    end = time.time()

    elapsedTime = end - start

    print('Done!')

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  

    parser.add_argument(  
      '--select',  
      type=str,  
      default='All',  
      help='select to run inference for our selector or all QFs '  
  )

    parser.add_argument(  
      '--conv',  
      type=str,
      default='../conv/',
      help='Feature Maps text directory directory'
  )

    parser.add_argument(  
      '--START',  
      type=int,  
      default='1',  
      help='start of the sequence  '  
  )

    parser.add_argument(  
      '--END',  
      type=int,  
      default='50001',  
      help='end of the sequence '  
  )
    parser.add_argument(  
      '--offset',  
      type=int,  
      default='0',  
      help='wnd of the sequence '  
  )


    parser.add_argument(  
      '--model_name',  
      type=str,  
      default='IV3',  
      help='model name'  
  )
    parser.add_argument(
      '--gen_dir',
      type=str,
      default='../gen/',
      help='generated directory'
  )

    parser.add_argument(  
      '--batch_size',  
      type=int,  
      default=1,  
      help='Recommended batch size is 100, so set it to 10 if you are running All CodeMode'  
  )

    FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  
