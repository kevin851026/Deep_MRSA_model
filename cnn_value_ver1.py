# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import json
def read_and_decode(filename_queue,number):
	# 建立 TFRecordReader
	reader = tf.TFRecordReader()

	# 讀取 TFRecords 的資料
	_, serialized_example = reader.read(filename_queue)

	# 讀取一筆 Example
	features = tf.parse_single_example(
	serialized_example,
	features={
	  'height': tf.FixedLenFeature([], tf.int64),
	  'width': tf.FixedLenFeature([], tf.int64),
	  'image_string': tf.FixedLenFeature([], tf.string),
	  'label': tf.VarLenFeature(tf.int64),
	  })
	# 將序列化的圖片轉為 uint8 的 tensor
	image = tf.decode_raw(features['image_string'], tf.uint8)
	label=[]
	# 將 label 的資料轉為 float32 的 tensor
	label =tf.cast(features['label'].values,tf.int32)

	# 將圖片的大小轉為 int32 的 tensor
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)

	# 將圖片調整成正確的尺寸
	image = tf.reshape(image, [height, width, 3])
	label=tf.reshape(label,[1])

	# 這裡可以進行其他的圖形轉換處理 ...
	# ...
	IMAGE_HEIGHT = 80
	IMAGE_WIDTH = 80
	# 圖片的標準尺寸
	#image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
	resized_image=tf.image.resize_images(image, size=(IMAGE_HEIGHT, IMAGE_WIDTH),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	# 將圖片調整為標準尺寸
	# resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
	# target_height=IMAGE_HEIGHT,
	# target_width=IMAGE_WIDTH)
	# resized_image=tf.image.per_image_standardization(resized_image)
	resized_image=tf.to_float(resized_image)

	# r,g,b=tf.split(resized_image,[1,1,1],2)

	# 打散資料順序

	images_batch, labels_batch = tf.train.shuffle_batch(
		[resized_image, label],
		batch_size=number, #這邊設BATCH SIZE
		capacity=3000,
		num_threads=1,
		min_after_dequeue=1000)
	

	return images_batch, labels_batch
def convNet(x_in,dropout,reuse,isTraining):
	with tf.variable_scope('cnn',reuse=reuse):
		inputLayer=tf.reshape(x_in,[-1,80,80,3])
		conv1=tf.layers.conv2d(inputs=inputLayer,filters=8,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu,reuse=None,name='conv1')
		# conv1_re=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu,reuse=True,name='conv1')
		pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])

		# conv2=tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu)
		# pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])

		conv3_1=tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[1,1],strides=1,padding='same',activation=tf.nn.relu)
		conv3_2=tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[3,3],strides=1,padding='same',activation=tf.nn.relu)
		# conv3_3=tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[5,5],strides=1,padding='same',activation=tf.nn.relu)
		pool3=tf.layers.max_pooling2d(inputs=pool1,pool_size=[3,3],strides=[2,2])
		# conv3=tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu)
		# pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])
		# print(pool2)
		flat_1=tf.reshape(conv3_1,[-1,40*40*16])
		flat_2=tf.reshape(conv3_2,[-1,40*40*16])
		# flat_3=tf.reshape(conv3_3,[-1,40*40*16])
		flat_4=tf.reshape(pool3,[-1,19*19*8])
		concat=tf.concat([flat_1, flat_2,flat_4], 1 )
		# inputLayer=tf.reshape(x_in,[-1,80*80])
		dense1=tf.layers.dense(inputs=concat, units=512, activation=tf.nn.relu)
		dropout1=tf.layers.dropout(inputs=dense1, rate=dropout, training=isTraining)
		dense2=tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
		dropout2=tf.layers.dropout(inputs=dense2, rate=dropout, training=isTraining)
#		dense3=tf.layers.dense(inputs=dropout2, units=2048, activation=tf.nn.relu)
#		dropout3=tf.layers.dropout(inputs=dense3, rate=dropout, training=isTraining)
#		dense4=tf.layers.dense(inputs=dropout3, units=2048, activation=tf.nn.relu)
#		dropout4=tf.layers.dropout(inputs=dense4, rate=dropout, training=isTraining)
		logits=tf.layers.dense(inputs=dropout2, units=1)

		#for test
		# logits=logits=tf.nn.softmax(logits) if not isTraining else logits

	return logits


if __name__=='__main__':
	#這邊放要讀入的 TF_RECORD
	tfrecord_filename=(["./value_record_2","./value_record_1","./value_record_0"])
	#tfrecord_filename=(["./train_tf_1","./train_tf_6","./train_tf_8"])
	#num_epoch 是上限epoch 數 設大一點
	filename_queue = tf.train.string_input_producer(tfrecord_filename, num_epochs=20000)
	images, labels = read_and_decode(filename_queue,140)

	test_tfrecord_filename=(["./new_val_record_0","./new_val_record_1"])
	test_filename_queue = tf.train.string_input_producer(test_tfrecord_filename, num_epochs=20000)
	test_images, test_labels = read_and_decode(test_filename_queue,2800)

	x_in=tf.placeholder(tf.float32, [None,80,80,3])
	y_in=tf.placeholder(tf.float32, [None,1])
	#for train
	trainGraph=convNet(x_in,0.7,False,True)
	#for test 
	# with tf.device('/cpu:0'):
	testGraph=convNet(x_in,1.,True,False)

	# oneHotLabels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)##################
	loss=tf.reduce_mean(tf.square(tf.divide(tf.abs(tf.subtract(trainGraph,y_in)),y_in)+1))
	test_loss=tf.reduce_mean(tf.square(tf.divide(tf.abs(tf.subtract(testGraph,y_in)),y_in)+1))
	# loss=tf.losses.mean_squared_error(labels=y_in,predictions=trainGraph)
	# test_loss=tf.losses.mean_squared_error(labels=y_in,predictions=trainGraph)
	# loss = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=trainGraph)
	# test_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=testGraph)
	optimizer=tf.train.AdamOptimizer(0.00001).minimize(loss)

	# lossT = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=testGraph)
	# optimizerT=tf.train.AdamOptimizer(0.001).minimize(lossT)

	# correct_prediction = tf.equal(tf.argmax(testGraph, 1), tf.argmax(y_in, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	#accuracy = tf.metrics.accuracy(labels=tf.argmax(y_in, axis=1), predictions=tf.argmax(trainGraph, axis=1))[1]
	init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	epoch=0
	chart_epoch=[]
	chart_train_loss=[]
	chart_val_loss=[]
	chart_train_acc=[]
	chart_val_acc=[]
	saveloss={'loss':[],'v_loss':[],'epoch':[]}
	with tf.Session()  as sess:
		# 初始化
		i=0
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		total=0
		saver=tf.train.Saver()
		model_name="1260/model1260"
		print(model_name)
		saver.restore(sess,model_name)
		while True:
			try:
				for op in tf.get_default_graph().get_operations():
					print( op.name)
				i+=1
				#trainImg,trainLabel=sess.run([images, labels])#向queue要一個batch的資料
				#sess.run(optimizer,feed_dict={x_in:trainImg,y_in:trainLabel})
				# x=sess.run(b,feed_dict={x_in:trainImg,y_in:trainLabel})
				# print(x.shape)
				#ls=sess.run(loss,feed_dict={x_in:trainImg,y_in:trainLabel})
				
				# total+=ls
				
				if i%1==0: # epoch 要自己算 epoch=全部/batch大小
					break
					testImg,testLabel=sess.run([test_images, test_labels])
					test_ls=sess.run(test_loss,feed_dict={x_in:testImg,y_in:testLabel})
					
					print('--------------------------------------')
					total=total/i
					
					print('Train loss = '+'{:.6f}'.format(total)+' Test  loss = '+'{:.6f}'.format(test_ls))
					epoch=epoch+1
					if epoch%5==0:

						chart_epoch.append(epoch)
						chart_train_loss.append(total)
						chart_val_loss.append(test_ls)
						saveloss['epoch'].append(epoch)
						saveloss['loss'].append(float(round(total,6)))
						saveloss['v_loss'].append(float(round(test_ls,6)))
						
					i=0
					total=0
					print(epoch)
					if epoch%20==0: #每50epoch 存一個
						plt.clf()
						plt.plot(chart_epoch,chart_train_loss,lw=3,label="training data")
						plt.plot(chart_epoch,chart_val_loss,"r",lw=3,label="validation data")
						plt.xlabel("epoch")
						plt.ylabel("loss")
						plt.legend(loc="best")
						plt.savefig('ver1_loss.jpg')
						saver = tf.train.Saver()
						savePath='./model_ver1/'+str(epoch)
						os.mkdir(savePath)#這行會噴 如果原本就有該資料夾的話
						mdlName='model'+str(epoch)
						saver.save(sess, os.path.join(savePath,mdlName))
						# print(saveloss)
						with open('loss.json','w') as f:
							json.dump(saveloss,f,indent=2)
						
			except tf.errors.OutOfRangeError:
				coord.request_stop()
				coord.join(threads)
				break

