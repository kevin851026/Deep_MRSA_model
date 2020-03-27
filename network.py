import tensorflow as tf
import numpy as np
import os

def getResizeData(img,label):
    # with tf.device('/cpu:0'):
    img=tf.read_file(img)
    img=tf.image.decode_image(img,channels=3)
    #img=tf.image.resize_image_with_crop_or_pad(img,200,400)
    img=tf.to_float(img)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    return img,label

def getData(direct,time):
    # with tf.device('/cpu:0'):
    images=[]
    labels=[]
    for i in os.listdir(direct):
        print(i)
        for j in os.listdir(direct+'/'+i):
            for file in os.listdir(direct+'/'+i+'/'+j+'/'+time):
                images.append(os.path.join(direct+'/'+i+'/'+j+'/'+time,file))
                if(i=='10'):
                    labels.append([1,0,0])
                elif(i=='100'):
                    labels.append([0,1,0])
                else:
                    labels.append([0,0,1])

    images=tf.constant(images)
    labels=tf.constant(labels)
    dataset=tf.data.Dataset.from_tensor_slices((images,labels))
    dataset=dataset.map(getResizeData)
    return dataset

def convNet(x_in,dropout,reuse,isTraining):
    with tf.variable_scope('cnn',reuse=reuse):
        inputLayer=tf.reshape(x_in,[-1,500,100,3])
        conv1=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[20,20],strides=2,padding='valid',activation=tf.nn.relu,reuse=None,name='conv1')
        # conv1_re=tf.layers.conv2d(inputs=inputLayer,filters=32,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu,reuse=True,name='conv1')
        pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=[2,2])

        conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[10,10],strides=1,padding='valid',activation=tf.nn.relu)
        pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,2])

        # conv3=tf.layers.conv2d(inputs=pool2,filters=200,kernel_size=[20,20],strides=2,padding='same',activation=tf.nn.relu)
        # pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],strides=[2,2])
        print(pool2)
        flat=tf.reshape(pool2,[-1,17600])

        dense1=tf.layers.dense(inputs=flat, units=600, activation=tf.nn.relu)
        dropout1=tf.layers.dropout(inputs=dense1, rate=dropout, training=isTraining)
        #dense2=tf.layers.dense(inputs=dropout1, units=600, activation=tf.nn.relu)
        #dropout2=tf.layers.dropout(inputs=dense2, rate=dropout, training=isTraining)
        logits=tf.layers.dense(inputs=dropout1, units=3)
        #for test
        logits=tf.nn.softmax(logits) if not isTraining else logits

    return logits





if __name__=='__main__':

    direct='./cut'
    time='10'
    trainSet=getData(direct,time)
    trainSet=trainSet.shuffle(buffer_size=100)
    trainSet=trainSet.batch(50)
    trainItrt = trainSet.make_initializable_iterator()
    trainData = trainItrt.get_next()

    x_in=tf.placeholder(tf.float32, [None,500,100,3])
    y_in=tf.placeholder(tf.float32, [None,3])
    #for train
    trainGraph=convNet(x_in,0.6,False,True)
    #for test 
    # with tf.device('/cpu:0'):
    testGraph=convNet(x_in,1.,True,False)

    # oneHotLabels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=4)##################
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=trainGraph)
    optimizer=tf.train.AdamOptimizer(0.001).minimize(loss)

    # lossT = tf.losses.softmax_cross_entropy(onehot_labels=y_in,logits=testGraph)
    # optimizerT=tf.train.AdamOptimizer(0.001).minimize(lossT)

    correct_prediction = tf.equal(tf.argmax(testGraph, 1), tf.argmax(y_in, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.Session()#config=tf.ConfigProto(log_device_placement=True)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print('--------------------------------------')
        print(i)
        sess.run(trainItrt.initializer)
        while True:
            try:
                trainImg,trainLabel=sess.run(trainData)
                sess.run(optimizer,feed_dict={x_in:trainImg,y_in:trainLabel})
                ls=sess.run(loss,feed_dict={x_in:trainImg,y_in:trainLabel})
                print('loss = '+'{:.6f}'.format(ls))
                train_acc=sess.run(accuracy,feed_dict={x_in:trainImg,y_in:trainLabel})
                print('train acc:'+'{:.4f}'.format(train_acc))
            except tf.errors.OutOfRangeError:
                break
        print('******************************************')