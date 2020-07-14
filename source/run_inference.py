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

import math 
import glob 
import time 
import imagenet
import sys
import argparse


from model_names import Models
from code_modes  import CodeMode
from conv_layer import Conv_Layer
from calc_space_methods import *
from sparsity_method_types import SparsityMethodTypes

# import imagenet_preprocessing as ipp
from preprocessing import inception_preprocessing, vgg_preprocessing
import pickle 

human_labels = imagenet.create_readable_names_for_imagenet_labels()
# np.set_printoptions(threshold=sys.maxsize, precision=3)
np.set_printoptions(threshold=sys.maxsize)

# Import all constants
from myconstants import *

def create_graph(): 
    with tf.gfile.FastGFile(MODEL_PATH + Frozen_Graph[FLAGS.model_name], 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def get_DNN_info(sess):

    graph_def = sess.graph.as_graph_def(add_shapes=True)

    all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
    all_layers  = []
    
    # loop on all nodes in the graph
    for  nid, n in enumerate(graph_def.node):
        try:
            
            #print(n.name)
            if 'Conv2D' in n.name:

                output_tensor_name = n.name + ':0'

                input_tensor_name = n.input[0] + ':0'
                input_tensor      = sess.graph.get_tensor_by_name(input_tensor_name)
                Ih                = input_tensor.shape[1]
                Iw                = input_tensor.shape[2]
                Ic                = input_tensor.shape[3]

                conv_tensor_params_name = n.input[1] + ':0'
                conv_tensor_params      = sess.graph.get_tensor_by_name(conv_tensor_params_name)
                filter_shape            = conv_tensor_params.shape
                Kh                      = filter_shape[0]
                Kw                      = filter_shape[1]
                K                       = filter_shape[3]
               
                Oh = n.attr['_output_shapes'].list.shape[0].dim[1].size
                Ow = n.attr['_output_shapes'].list.shape[0].dim[2].size

                if 'padding' in n.attr.keys():
                    padding_type = n.attr['padding'].s.decode(encoding='utf-8')
                if 'strides' in n.attr.keys():
                    art_tensor_name = n.name + ':0'
                    strides_list = [int(a) for a in n.attr['strides'].list.i]
                    Sh           = strides_list[1]
                    Sw           = strides_list[2]
                
                # Create the conv layer
                conv_layer = Conv_Layer(input_tensor_name, output_tensor_name, K, Kh, Kw, Sh, Sw, Oh, Ow, Ih, Iw, Ic, In=1, padding=padding_type)
                all_layers.append(conv_layer)
        except ValueError:
            print('%s is an Op.' % n.name)

    return all_layers
    
def get_DNN_info_pld(sess):

    graph_def = sess.graph.as_graph_def(add_shapes=True)

    all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
    all_layers  = []
    for itensor, tensor in enumerate(all_tensors):
        #print(tensor)
        if 'conv2d_params' in tensor.name:
            filter_shape = tensor.shape
            Kh           = filter_shape[0]
            Kw           = filter_shape[1]
            K            = filter_shape[3]
            
            input_tensor = all_tensors[itensor-1]
            input_tensor_name = input_tensor.name
            Ih = input_tensor.shape[1]
            Iw = input_tensor.shape[2]
            Ic = input_tensor.shape[3]

        if 'Conv2D' in tensor.name:
            output_shape = tensor.shape
            Oh           = output_shape[1]
            Ow           = output_shape[2]
            output_tensor_name = tensor.name
            
            Sh = -1
            Sw = -1
            
            conv_layer = Conv_Layer(input_tensor_name, output_tensor_name, K, Kh, Kw, Sh, Sw, Oh, Ow, Ih, Iw, Ic)
            all_layers.append(conv_layer)

            #print(conv_layer, ' Layer Count: ', len(all_layers), conv_layer.output_tensor_name)
    
    #print('\n\n\nDNN Law Yoom ye3ady')

    layer_count = 0

    for  nid, n in enumerate(graph_def.node):
        try:
            
            #print(n.name)
            if 'Conv2D' in n.name:
                if 'padding' in n.attr.keys():
                    padding_type = n.attr['padding'].s.decode(encoding='utf-8')
                    all_layers[layer_count].padding = padding_type
                if 'strides' in n.attr.keys():
                    art_tensor_name = n.name + ':0'
                    strides_list = [int(a) for a in n.attr['strides'].list.i]
                    Sh           = strides_list[1]
                    Sw           = strides_list[2]
                    
                    # Update the stride information
                    #print('Layer Count: ', layer_count, n.name)
                    all_layers[layer_count].Sh = Sh
                    all_layers[layer_count].Sw = Sw
                    all_layers[layer_count].padding_cal()
                    layer_count = layer_count + 1
                    #print (n.name, ' Strides: ' , [int(a) for a in n.attr['strides'].list.i])
        except ValueError:
            print('%s is an Op.' % n.name)
    return all_layers

# ---- 

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


##################### May 6, 2019 -- with Excel sheet ####################


#vgg_synset = [l.strip() for l in open(vgg_synset_file_path).readlines()]
def run_predictions(sess, image_batch, softmax_tensor, startBatch, qf_idx_list, how_many_qfs, sheet, style, gt_label_list):

    if models[FLAGS.model_name] != Models.inceptionresnetv2.value:
        bot = sess.graph.get_tensor_by_name('final_layer/Mean:0')
        predictions, bot_value = sess.run([softmax_tensor,bot ], {'input:0': sess.run(image_batch)})
        print(bot_value)
        exit()
        my_batch_size = image_batch.shape[0]
    else:
        predictions = sess.run(softmax_tensor, {'image:0': sess.run(image_batch)})
        my_batch_size = 1

    predictions = np.squeeze(predictions)
   
    # for index, value in enumerate(predictions):
    #     print(index, value, human_labels[index])

    N             = -1000
    
    for img in range(my_batch_size):
        
        current_rank  = -1

        if my_batch_size > 1 or FLAGS.select == CodeMode.getCodeName(3):
            predictions_img = predictions[img]
        else:
            predictions_img = predictions

        top_5 = predictions_img.argsort()[N:][::-1]

        for rank, node_id in enumerate(top_5):

           if models[FLAGS.model_name] <= Models.inceptionresnetv2.value:
            human_string = human_labels[node_id]
           else:
            human_string = vgg_synset[node_id]
            human_string = human_string.split(" ", 1)
            human_string = human_string[1]

        

           score = predictions_img[node_id] 
           print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))
            
           # Update the id every interval in the original case
           if FLAGS.select == CodeMode.getCodeName(3): 
            if (img % how_many_qfs == 0):
                idx = (img // how_many_qfs) + startBatch
           else:
            idx = img + startBatch

           #print(img, how_many_qfs, (img % how_many_qfs), idx)
           #print(human_string, gt_label_list[idx], 'Helloooooo')
           if(gt_label_list[idx] == human_string):
            print(node_id)
            print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))
            # Write the rank and the score
            # row = img + startBatch
            row = idx
            if FLAGS.select == CodeMode.getCodeName(1):
                col = 4
            if FLAGS.select == CodeMode.getCodeName(2):
                col = 2
            if FLAGS.select ==  CodeMode.getCodeName(3):
                col = 6 + 2*qf_idx_list[img]

            # Set the current rank (rank starts from 0)
            current_rank = 1 + rank
            # print(human_string)
            # print(current_rank)

            #print(row, col)
            sheet.write(row, col, current_rank, style)
            sheet.write(row, 1 + col, score.item(), style)
                
            # Stop looping once you find it in the rank
            break

    return 0


def cal_densityBound(layer):

    print('\n------ Im2col vs CPO-------\n')

    S1 = layer.Ow*layer.Kw/layer.Sw + layer.Kw/layer.Sw + layer.Ow + 1 + 2*layer.ru*layer.Ih_padded*layer.Iw
    S2 = layer.Ow*layer.Kw*layer.Ih
    
    if (layer.Kw%layer.Sw) == 0:
        S1_cmp = (layer.Kw/layer.Sw)*(layer.Ow+1)+2*layer.Ih*layer.Iw*layer.ru # we should multiply by Ic here, create seperate functions for this
    elif (layer.Kw%layer.Sw) != 0:
        S1_cmp = math.ceil(layer.Kw/layer.Sw)*(layer.Ow+1)+2*layer.Ih*layer.Iw*layer.ru
        
    S_im2col = (math.ceil((layer.Iw-layer.Kw)/layer.Sw)+1)*(math.ceil((layer.Ih-layer.Kh)/layer.Sh)+1)*layer.Kw*layer.Kh
    print(('S_Im2col (S4) : %f')% (S_im2col))
    print(('Im2col vs CPO S4-S1 : %f ')% (S_im2col-S1))
    print(("Compression Ratio (CPO vs Im2col): %.2fx")%(S_im2col/S1_cmp))

    print('------ MEC vs CPO -------\n')
    #MEC - CPO
    S_mec_cop = S2-S1
    print(("MEC (S2) = %f  && CPO (S1) = %f")%(S2,S1))
    
    # print('Value of ru = %f , S1 = %f , S2 = %f, S2-S1= %f'%(ru, S1, S2, S_mec_cop))
    density_bound_mec = (layer.Ow*layer.Kw*layer.Ih - (layer.Ow*layer.Kw)/layer.Sw - layer.Kw/layer.Sw - layer.Ow - 1)
    density_bound_mec = density_bound_mec / (2*layer.Ih*layer.Iw)

    print(('MEC vs CPO S2-S1 : %f || with Feature_maps density = %f || Density bound MEC vs. CPO = %f')% (S_mec_cop, layer.ru, density_bound_mec))
    print(("Compression Ratio (MEC vs Im2col): %.2fx")%(S_im2col/S2))
    
    #######

    print('\n------ CSCC vs CPO-------\n')

    # density_lowering = max(lowering_desity_channel)
    # CSCC - CPO
    term1 = layer.Ow*layer.density_lowering
    term2 = (layer.Ow+1)/(2*layer.Ih*layer.Sw)
    term0 = (layer.Kw/layer.Iw)
    # print(term0,term1,term2)
    density_bound_cscc = term0*(term1 - term2)

    term1 = (math.ceil((layer.Iw-layer.Kw)/layer.Sw)+2)
    term2 = 2*(math.ceil((layer.Iw-layer.Kw)/layer.Sw)+1)*layer.Kw*layer.Ih*layer.density_lowering
    S_cscc = term1 + term2 #S3
    
    term0 = (2*layer.Ow*layer.Ih*layer.Kw*layer.density_lowering)
    tetm1 = (2*layer.Ih*layer.Iw*layer.ru)
    term2 = (layer.Kw/layer.Sw)*(layer.Ow+1)
    S_cscc_cpo = term0 - term1 - term2

    print(('S_cscc (S3) : %f')% (S_cscc))
    print(('CSCC vs CPO S3-S1 : %f || with Lowering density = %f || Density bound CSCC vs. CPO = %f')% (S_cscc_cpo, layer.density_lowering, density_bound_cscc))
    print(("Compression Ratio (CSCC vs Im2col): %.2fx")%(S_im2col/S_cscc))


    print('\n------ END of Analysis-------\n')
    
    return (density_bound_mec, density_bound_cscc)

def overlap_cal(lowering_matrix, kw ,kh , sw, sh , tot_nz_feature):
    kw = int(kw)
    kh = int(kh)
    sw = int(sw)
    sh = int(sh)

    print('Patterns Calculations')
    tot_nz_lowering = np.size(lowering_matrix[lowering_matrix != 0.0])
    # SET CAL
    if (kw%sw)==0:
        num_ptr = math.floor(kw/sw)
    else:
        num_ptr = math.floor(kw/sw) + 1
    
    pattern_set = np.zeros(num_ptr)
    for pattern_idx in range(num_ptr,1,-1):
        for idx in range(0,np.shape(lowering_matrix)[0]):
            # idx+num_ptr-1   should be a variable that decrement
            if ((idx+pattern_idx-1)<np.shape(lowering_matrix)[0]):
                row_ptr = np.empty((0,np.shape(lowering_matrix)[1]), int)

                for row_idx in range(0,pattern_idx):
                    x = np.roll(lowering_matrix[idx+row_idx,:],row_idx)
                    x[x==0] = 256 + row_idx
                    x = np.reshape(x,(1,np.size(x)))
                    row_ptr = np.append(row_ptr, x, axis=0)
                acum = np.ones(np.shape(row_ptr)[1], dtype=bool)
                            
                for row_u in range(0,pattern_idx):
                    for row_l in range(row_u+1,pattern_idx):
                        x = (row_ptr[row_u,:]==row_ptr[row_l,:])
                        acum = acum & x
                
                set_count = np.sum(acum)
                pattern_set[pattern_idx-1] = pattern_set[pattern_idx-1] + set_count
                if (set_count>0):
                    for row_idx in range(0,pattern_idx):
                        x = np.roll(acum,-1*row_idx)
                        y = lowering_matrix[idx+row_idx,:]
                        y[x] = 0
                        lowering_matrix[idx+row_idx,:] = y
    pattern_idx = 1
    pattern_set[pattern_idx-1] = (np.size(lowering_matrix[lowering_matrix != 0.0]))
    
    pattern_set_perc = 100*pattern_set/tot_nz_feature
    
    for pattern_idx in range(0,np.size(pattern_set)):
        print(("Counts of Set #%d --> %f ")%(pattern_idx+1,pattern_set_perc[pattern_idx]))

    # Checks 
    tot_pattern = 0
    for pattern_idx in range(0,np.size(pattern_set)):
        tot_pattern = tot_pattern + (pattern_idx+1)*pattern_set[pattern_idx]


    print("Total Number of Non-Zero after creating the patterns: %d"%tot_pattern)
    print("Total Number of Non-Zero of the lowering matrix: %d"%tot_nz_lowering)
    if (tot_pattern!=tot_nz_lowering):
        print("ERROR: missing some patterns")
        exit(0)
    print('\n-------------\n')
    return pattern_set_perc

def featureMap_stats(feature_maps, layer):

    feature_maps = layer.image_padding(feature_maps)
    lowering_matrix = layer.lowering_rep(feature_maps)
    layer.cal_density(lowering_matrix)
    print(getSpaceCPO(layer))
    print(getDensityBoundLoweringDensityCSCC(layer))
    for key in conv_methods:
        if (conv_methods[key] == SparsityMethodTypes.CPO.value):
            pass
        elif (conv_methods[key] == SparsityMethodTypes.CPS.value):
            pass
        elif (conv_methods[key] == SparsityMethodTypes.MEC.value):
            pass
        elif (conv_methods[key] == SparsityMethodTypes.CSCC.value):
            pass
        elif (conv_methods[key] == SparsityMethodTypes.Im2Col.value):
            pass
    # density_bound_mec, density_bound_cscc = cal_densityBound(layer)

    # assert(models[FLAGS.model_name] == Models.inceptionresnetv2.value or models[FLAGS.model_name] == Models.IV3.value)


    # print(("The bound (MEC)-(CPO) : %f && The bound (CSCC)-(CPO) : %f")%(density_bound_mec, density_bound_cscc))
    # print('\n-------------\n')
    return lowering_matrix, layer.tot_nz_feature

def run_predictionsImage(sess, image_data, softmax_tensor, idx, qf_idx):
    # Input the image, obtain the softmax prob value（one shape=(1,1008) vector）
    # predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data}) # n, m, 3

    # Only used for InceptionResnetV2
    assert(models[FLAGS.model_name] == Models.inceptionresnetv2.value or models[FLAGS.model_name] == Models.IV3.value)

    all_layers = get_DNN_info(sess)

    #for l in all_layers:
    #    print(l)
    # exit(0)

    input_tensor = 'image:0'
    if models[FLAGS.model_name] == Models.IV3.value:
        input_tensor_name = 'DecodeJpeg/contents:0'
        #predictions = sess.run(softmax_tensor, {input_tensor_name: image_data})
    else:
        predictions = sess.run(softmax_tensor, {input_tensor_name: sess.run(image_data)})

    # count_Kw_Kh = 0
    # for layer in all_layers:
    #     if (layer.Kw == 1 and layer.Kh == 1):
    #         count_Kw_Kh = count_Kw_Kh + 1
    # print("kw = 1 , kh = 1 counts", count_Kw_Kh)
    # for layer in all_layers:
    print(np.shape(all_layers))
    layer = all_layers[3]
    print(layer)
    current_tensor      = sess.graph.get_tensor_by_name(layer.input_tensor_name)
    current_feature_map = sess.run(current_tensor, {input_tensor_name: image_data})
    # print(current_feature_map.shape)
    current_feature_map = np.squeeze(current_feature_map)
    # lowering_matrix, tot_nz_feature = feature_analysis(current_feature_map, layer.padding, layer.Kw ,layer.Kh , layer.Sw, layer.Sh, layer.Ow, layer.Oh )
    lowering_matrix, tot_nz_feature = featureMap_stats(current_feature_map, layer)
    CPO = overlap_cal(lowering_matrix, layer.Kw, layer.Kh , layer.Sw, layer.Sh , tot_nz_feature )
    print(layer)
    exit(0)

    #relu_conv_tensor = sess.graph.get_tensor_by_name('mixed_' + str(layerID) + '/join:0')

    return 1

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



def readImageBatch(startBatch, endBatch, qf_list):
    
    batch_size     = endBatch - startBatch + 1
    nChannels      = 3

    # inceptionresnetv2 does not support batching
    if models[FLAGS.model_name] != Models.inceptionresnetv2.value:
        img_batch      = tf.zeros([0, resized_dimention[FLAGS.model_name], resized_dimention[FLAGS.model_name], nChannels], dtype=tf.float32)
 
    qf_idx_list    = []

    for imgID in range(startBatch, endBatch):
        original_img_ID         = imgID
        imgID                   = str(imgID).zfill(8)
        shard_num               = math.ceil(original_img_ID/10000) -1
        folder_num              = math.ceil(original_img_ID/1000)
        
        if FLAGS.select == CodeMode.getCodeName(2): # Org
            # Read the original
            current_jpeg_image      = org_image_dir + '/shard-' + str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '.JPEG'
            #current_jpeg_image      = '/home/h2amer/work/workspace/ML_TS/training_original/shard-9/944/n02834397_3465.JPEG'
            image_data              = readImage(current_jpeg_image)
            
            if models[FLAGS.model_name] == Models.inceptionresnetv2.value:
                img_batch               = image_data
            else:
                img_batch               = tf.concat([img_batch, image_data], 0)
        
        if FLAGS.select == CodeMode.getCodeName(3): # All
            for qf_idx, qf in enumerate(qf_list):
                current_jpeg_image      = image_dir + '/shard-'  + str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' \
                        + imgID + '-QF-' + str(qf) + '.JPEG'
                image_data              = readImage(current_jpeg_image)
                
                if models[FLAGS.model_name] == Models.inceptionresnetv2.value:
                    img_batch               = image_data
                else:
                    img_batch               = tf.concat([img_batch, image_data], 0)
                qf_idx_list.append(qf_idx)        

    return img_batch, qf_idx_list


def readAndPredictOptimizedImageByImage():

    # Only used for InceptionResnetV2 or IV3
    assert(models[FLAGS.model_name] == Models.inceptionresnetv2.value or (models[FLAGS.model_name] == Models.IV3.value))


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

               
        if original_img_ID < 0: ## till 48000 shoule generate again
            continue
       
        else:
            imgID = str(imgID).zfill(8)
            shard_num = math.ceil(original_img_ID/10000) -1 
            folder_num = math.ceil(original_img_ID/1000) 

            
            if FLAGS.select == CodeMode.getCodeName(1): # Org
                qf_idx     =  0
                qf         = 110
                current_jpeg_image      = org_image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '.JPEG'
                
                if (FLAGS.model_name == Models.inceptionresnetv2.value):
                    image_data = tf.read_file(current_jpeg_image)
                    image_data = tf.image.decode_jpeg(image_data, channels=3)
                else:
                    image_data = tf.gfile.FastGFile(current_jpeg_image, 'rb').read()
                run_predictionsImage(sess, image_data, softmax_tensor, actual_idx, qf_idx)
            
            else:
                for qf_idx, qf in enumerate(qf_list) :
                    if qf == 110 :
                        current_jpeg_image      = org_image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '.JPEG'
                    else :
                        current_jpeg_image      = image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '-QF-' + str(qf) + '.JPEG'
                    
                    image_data = tf.read_file(current_jpeg_image)
                    image_data = tf.image.decode_jpeg(image_data, channels=3)
                    run_predictionsImage(sess, image_data, softmax_tensor, actual_idx, qf_idx)

        if (actual_idx) % 50 == 0:
                tf.reset_default_graph()
                sess.close()
        

        t += time.time() - startTime
        if not original_img_ID % 10 :
            print ('image %d is done in %f seconds' % (original_img_ID, t))
            t = 0

    print('Final Save...')

    return top1_acc , top5_acc 

def readAndPredictLoopOptimized():
    
    # Get the ground truth list
    qf_list  = construct_qf_list()
    img_size   = resized_dimention[FLAGS.model_name]
    t        = 0
    top1_acc = 0 
    top5_acc = 0 

    num_batches = int(math.ceil(examples_count/FLAGS.batch_size))

    for batchID in range(num_batches):

        startBatch               = FLAGS.START + FLAGS.offset + batchID*FLAGS.batch_size
        endBatch                 = startBatch  + FLAGS.batch_size 
        image_batch, qf_idx_list = readImageBatch(startBatch, endBatch, qf_list)
        startTime                = time.time()
        


        # if (batchID) % 10 == 0 :
            
        config = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config)
        create_graph()
        #print_tensors(sess)
        get_DNN_info(sess)
        exit(0)
        
        softmax_tensor = sess.graph.get_tensor_by_name(final_tensor_names[FLAGS.model_name])
        input_tens = sess.graph.get_tensor_by_name('input:0')
       
        print('New session group has been created')

        
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        # sess = tf.Session(config=config)
        # create_graph()
        # softmax_tensor = sess.graph.get_tensor_by_name(final_tensor_names[FLAGS.model_name])
        run_predictions(sess, image_batch, softmax_tensor, startBatch, qf_idx_list, len(qf_list), sheet, style, gt_label_list)
        
        t += time.time() - startTime
        #if not batchID % 2:
        print ('Batch %d/%d of size %d is done in %f seconds.' % (1 + batchID, num_batches, FLAGS.batch_size, t))
        t = 0


        # if (batchID + 1 ) % 10 == 0:
        tf.reset_default_graph()
        sess.close()
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
# python3-tf run_inference.py --select Org --model_name IV3
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
    
    if models[FLAGS.model_name] == Models.inceptionresnetv2.value or models[FLAGS.model_name] == Models.IV3.value:
        top1_count, top5_count = readAndPredictOptimizedImageByImage()
    else:
        top1_count, top5_count = readAndPredictLoopOptimized()
    top1_accuracy = top1_count / num_images *100 
    top5_accuracy = top5_count / num_images *100 
    print('top1_accuracy == ', top1_accuracy ,'top5_accuracy == ',  top5_accuracy )

    end = time.time()

    elapsedTime = end - start

    print('Done in %s' % PATH_TO_EXCEL)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  

    parser.add_argument(  
      '--select',  
      type=str,  
      default='All',  
      help='select to run inference for our selector or all QFs '  
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
      '--batch_size',  
      type=int,  
      default=1,  
      help='Recommended batch size is 100, so set it to 10 if you are running All CodeMode'  
  )

    FLAGS, unparsed = parser.parse_known_args()
    print("\n############ Here WTF ############\n")
    print(FLAGS.model_name)
    print(SparsityMethodTypes.getModelByValue(1))
    print(SparsityMethodTypes.getModelByName('CPO'))
    print("\n############ END  WTF ############\n")

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  