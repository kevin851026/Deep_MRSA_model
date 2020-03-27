# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import random
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
	resized_image=tf.image.per_image_standardization(resized_image)
	resized_image=tf.to_float(resized_image)

	r,g,b=tf.split(resized_image,[1,1,1],2)

	# 打散資料順序

	images_batch, labels_batch = tf.train.shuffle_batch(
		[resized_image, label],
		batch_size=number, #這邊設BATCH SIZE
		capacity=3000,
		num_threads=2,
		min_after_dequeue=2000)
	

	return images_batch, labels_batch
def convNet(x_in,dropout,reuse,isTraining):
	with tf.variable_scope('cnn',reuse=reuse):
		# inputLayer=tf.reshape(x_in,[-1,80,80,3])
		# conv1=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu,reuse=None,name='conv1')
		# # conv1_re=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu,reuse=True,name='conv1')
		# pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])

		# conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu)
		# pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])

		# # conv3=tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[10,10],strides=1,padding='same',activation=tf.nn.relu)
		# # pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])
		# # print(pool2)
		# flat=tf.reshape(pool2,[-1,20*20*64])

		inputLayer=tf.reshape(x_in,[-1,80*80])

		dense1=tf.layers.dense(inputs=inputLayer, units=4096, activation=tf.nn.relu)
		dropout1=tf.layers.dropout(inputs=dense1, rate=dropout, training=isTraining)
		dense2=tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
		dropout2=tf.layers.dropout(inputs=dense2, rate=dropout, training=isTraining)
		dense3=tf.layers.dense(inputs=dropout2, units=5120, activation=tf.nn.relu)
		dropout3=tf.layers.dropout(inputs=dense3, rate=dropout, training=isTraining)
		dense4=tf.layers.dense(inputs=dropout3, units=2048, activation=tf.nn.relu)
		dropout4=tf.layers.dropout(inputs=dense4, rate=dropout, training=isTraining)
		dense5=tf.layers.dense(inputs=dropout4, units=2048, activation=tf.nn.relu)
		dropout5=tf.layers.dropout(inputs=dense5, rate=dropout, training=isTraining)
		dense6=tf.layers.dense(inputs=dropout5, units=512, activation=tf.nn.relu)
		dropout6=tf.layers.dropout(inputs=dense6, rate=dropout, training=isTraining)
		
		out=tf.layers.dense(inputs=dropout6, units=1)
		#for test
		# logits=tf.nn.softmax(logits) if not isTraining else logits

	return out


if __name__=='__main__':
	#這邊放要讀入的 TF_RECORD
	# tfrecord_filename=(["./train_tf_0","./train_tf_1","./train_tf_2","./train_tf_3","./train_tf_4","./train_tf_5",
	# 	"./train_tf_6","./train_tf_7","./train_tf_8"])
	# tfrecord_filename=(["./train_tf_2","./train_tf_3","./train_tf_5","./train_tf_6","./train_tf_7","./train_tf_8"])
	# #num_epoch 是上限epoch 數 設大一點
	# filename_queue = tf.train.string_input_producer(tfrecord_filename, num_epochs=1)
	# images, labels = read_and_decode(filename_queue,1)

	# test_tfrecord_filename=(["./test_tf_0","./test_tf_1","./test_tf_2","./test_tf_3","./test_tf_4"])
	# test_filename_queue = tf.train.string_input_producer(test_tfrecord_filename, num_epochs=200000)
	# test_images, test_labels = read_and_decode(test_filename_queue,6300)

	x_in=tf.placeholder(tf.float32, [None,80,80,1])
	y_in=tf.placeholder(tf.float32, [None,1])
	#for train
	trainGraph=convNet(x_in,0.8,False,True)
	#for test 
	# with tf.device('/cpu:0'):
	testGraph=convNet(x_in,1.,True,False)

	# oneHotLabels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)##################
	loss=tf.reduce_mean(tf.square(tf.divide(tf.abs(tf.subtract(trainGraph,y_in))+1,y_in+1)+1))
	test_loss=tf.reduce_mean(tf.square(tf.divide(tf.abs(tf.subtract(testGraph,y_in))+1,y_in+1)+1))
	# loss = tf.reduce_mean(tf.square(tf.subtract(testGraph, y_in)))
	# test_loss=tf.losses.absolute_difference(labels=y_in,predictions=testGraph)
	optimizer=tf.train.AdamOptimizer(0.00001).minimize(loss)

	# lossT = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=testGraph)
	# optimizerT=tf.train.AdamOptimizer(0.001).minimize(lossT)

	# correct_prediction = tf.equal(tf.argmax(testGraph, 1), tf.argmax(y_in, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# accuracy = tf.metrics.accuracy(labels=tf.argmax(y_in, axis=1), predictions=tf.argmax(trainGraph, axis=1))[1]
	init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	epoch=0
	with tf.Session()  as sess:
		# 初始化
		i=0
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess,coord=coord)
		total=0
		saver=tf.train.Saver()
		model_name="300/model300"
		print(model_name)
		saver.restore(sess,model_name)
		while True:
			try:
				i+=1
# 				trainImg,trainLabel=sess.run([images, labels])#向queue要一個batch的資料
# 				predict=sess.run(testGraph,feed_dict={x_in:trainImg,y_in:trainLabel})
# #				print(trainLabel.shape)
# 				print(predict[0][0],' ',trainLabel[0][0])
#				sess.run(optimizer,feed_dict={x_in:trainImg,y_in:trainLabel})
#				# ls=sess.run(loss,feed_dict={x_in:trainImg,y_in:trainLabel})
#				# print('loss = '+'{:.6f}'.format(ls))
#				ls=sess.run(loss,feed_dict={x_in:trainImg,y_in:trainLabel})
				# print('loss = '+'{:.6f}'.format(ls))
#				total+=ls
				if i%1==0: # epoch 要自己算 epoch=全部/batch大小
					break
					# testImg,testLabel=sess.run([test_images, test_labels])
					# test_ls=sess.run(test_loss,feed_dict={x_in:testImg,y_in:testLabel})
					# print('--------------------------------------')
					# total=total/i
					# print('loss = '+'{:.6f}'.format(total)+'  test = '+'{:.6f}'.format(test_ls))
					# epoch=epoch+1
					# i=0
					# total=0
					# print(epoch)
					# if epoch%50==0: #每50epoch 存一個
					# 	saver = tf.train.Saver()
					# 	savePath='./model/'+str(epoch)
					# 	os.mkdir(savePath)#這行會噴 如果原本就有該資料夾的話
					# 	mdlName='model'+str(epoch)
					# 	saver.save(sess, os.path.join(savePath,mdlName))
			except tf.errors.OutOfRangeError:
				coord.request_stop()
				coord.join(threads)
				break

