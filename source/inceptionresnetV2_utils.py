# TBCompleted

def run_predictionsImage(sess, image_data, softmax_tensor, idx, qf_idx, sheet, style, gt_label_list):
    # Input the image, obtain the softmax prob value（one shape=(1,1008) vector）
    # predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data}) # n, m, 3

    # Only used for InceptionResnetV2
    assert(models[FLAGS.model_name] == Models.inceptionresnetv2.value)

    predictions = sess.run(softmax_tensor, {'image:0': sess.run(image_data)}) 

    # (1, 1008)->(1008,)
    predictions = np.squeeze(predictions)
    N = -1000

    # Current_rank = -1
    current_rank = -1

    top_5 = predictions.argsort()[N:][::-1]
    for rank, node_id in enumerate(top_5):
        # print('node id ' , node_id)
        human_string = human_labels[node_id]
        score = predictions[node_id] 
        # print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))

        if(gt_label_list[idx] == human_string):
            # print('%d: %s (score = %.5f)' % (1 + rank, human_string, score))
            # Write the rank and the score
        
            row = idx

            if FLAGS.select == CodeMode.getCodeName(1):
                col = 4
            if FLAGS.select == CodeMode.getCodeName(2):
                col = 2
            if FLAGS.select ==  CodeMode.getCodeName(3):
                col = 6 + 2*qf_idx
            
            
            # Set the current rank (rank starts from 0)
            current_rank = 1 + rank
            # print(human_string)
            # print(current_rank)
            

            # print(row, col)
            sheet.write(row, col, current_rank, style)
            sheet.write(row, 1 + col, score.item(), style)
            
            # Stop looping once you find it in the rank
            break

    return current_rank



def readAndPredictOptimizedImageByImage(PATH_TO_EXCEL):

    # Only used for InceptionResnetV2
    assert(models[FLAGS.model_name] == Models.inceptionresnetv2.value)

    # Create the excel sheet workbook
    path_to_excel = PATH_TO_EXCEL
    rb = xlrd.open_workbook(path_to_excel, formatting_info=True)
    wb = copy(rb)
    sheet = wb.get_sheet(0)
    style = XFStyle()
    style.num_format_str = 'general'

    print('Done excel sheet creation')

    # Get the ground truth list
    sheet_r       = rb.sheets()[0]
    gt_label_list = sheet_r.col_values(1)
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

               
        #if original_img_ID ==  47591 or original_img_ID == 47592: ## till 48000 shoule generate again
        if original_img_ID < 0: ## till 48000 shoule generate again
            continue
       
        else:
            imgID = str(imgID).zfill(8)
            shard_num = math.ceil(original_img_ID/10000) -1 
            folder_num = math.ceil(original_img_ID/1000) 

            
            if FLAGS.select == CodeMode.getCodeName(1): # Selector
                qf_idx = 0 
                qf  =  selector_qf[original_img_ID -1]

                current_jpeg_image      = image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '-QF-' + str(qf) + '.JPEG'
                image_data = tf.read_file(current_jpeg_image)
                image_data = tf.image.decode_jpeg(image_data, channels=3)
                run_predictionsImage(sess, image_data, softmax_tensor, actual_idx, qf_idx, sheet, style, gt_label_list)
            
            elif FLAGS.select == CodeMode.getCodeName(2): # Org
                qf_idx     =  0
                qf         = 110
                current_jpeg_image      = org_image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '.JPEG'
                image_data = tf.read_file(current_jpeg_image)
                image_data = tf.image.decode_jpeg(image_data, channels=3)
                run_predictionsImage(sess, image_data, softmax_tensor, actual_idx, qf_idx, sheet, style, gt_label_list)
            
            else:
                for qf_idx, qf in enumerate(qf_list) :
                    if qf == 110 :
                        current_jpeg_image      = org_image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '.JPEG'
                    else :
                        current_jpeg_image      = image_dir + '/shard-' +str(shard_num) + '/' +  str(folder_num) + '/' + 'ILSVRC2012_val_' + imgID + '-QF-' + str(qf) + '.JPEG'
                    
                    image_data = tf.read_file(current_jpeg_image)
                    image_data = tf.image.decode_jpeg(image_data, channels=3)
                    run_predictionsImage(sess, image_data, softmax_tensor, actual_idx, qf_idx, sheet, style, gt_label_list)

        if (actual_idx) % 50 == 0:
                tf.reset_default_graph()
                sess.close()
        

        t += time.time() - startTime
        if not original_img_ID % 10 :
            print ('image %d is done in %f seconds' % (original_img_ID, t))
            t = 0

        if original_img_ID == 30:
            wb.save(path_to_excel)

        if original_img_ID == FLAGS.START + 10:
            wb.save(path_to_excel)

        if not original_img_ID % 1000:
            wb.save(path_to_excel)
    print('Final Save...')
    wb.save(path_to_excel)

    

    return top1_acc , top5_acc 