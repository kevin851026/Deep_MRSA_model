# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from skimage import io,transform
import random
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tfRecord_cread(direct,tfrecord_filename):

    drct=''
    images=[]
    labels=[]
    for i in ['1fg-uL','5fg-uL','50fg-uL','100fg-uL']:
            #    0        100      500       1000      5000     10000      50000
        drct=direct+i
        for file in os.listdir(drct):
            images.append(os.path.join(drct,file))
            if i=='1fg-uL':
#               labels.append([random.randint(49,51)])
                labels.append([1,0,0,0])
            elif i=='5fg-uL':
#                labels.append([random.randint(99,101)])
                labels.append([0,1,0,0])
#            elif i=='10fg-uL':
##                labels.append([random.randint(498,502)])
#                labels.append([0,0,1,0,0])
            elif i=='50fg-uL':
#                labels.append([random.randint(998,1002)])
                labels.append([0,0,1,0])
            elif i=='100fg-uL':
#                labels.append([random.randint(4998,5002)])
                labels.append([0,0,0,1])
#            else:
##                labels.append([random.randint(49998,50002)])
#                labels.append([0,0,0,0,0,0,1])
    count=0
    for i in range(8):#這個迴圈看你要做幾個檔案出來 現在是每X個會存一個檔案 
        writer = tf.python_io.TFRecordWriter(tfrecord_filename+str(i))
        try:
            while True:
                image_filename=images.pop()
                label=labels.pop()
                count=count+1
                image = io.imread(image_filename)
                print(label)
                #也可以在這邊再下標籤
                #label=XXXX
                #
                #
                #
                #
                height, width, depth = image.shape
                image_string = image.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature([height]),
                    'width': _int64_feature([width]),
                    'image_string': _bytes_feature(image_string),
                    'label': _int64_feature(label),
                    }))#
                writer.write(example.SerializeToString())
                print(image_filename)
                print(count)
                if count%2000==0:
                    break
                # elif i==2 and count==4151:
                #     break
            writer.close()
        except:
            writer.close()
            break

def tfRecord_test(file_name):#用這個函式確定一下自己有正確標籤
    record_iterator = tf.python_io.tf_record_iterator(path=file_name)
    for string_record in record_iterator:
        # 建立 Example
        example = tf.train.Example()

        # 解析來自於 TFRecords 檔案的資料
        example.ParseFromString(string_record)

        # 取出 height 這個 Feature
        height = int(example.features.feature['height']
                                   .int64_list
                                   .value[0])

        # 取出 width 這個 Feature
        width = int(example.features.feature['width']
                                  .int64_list
                                  .value[0])

        # 取出 image_string 這個 Feature
        image_string = (example.features.feature['image_string']
                                    .bytes_list
                                    .value[0])
        # 取出 label 這個 Feature
        label = (example.features.feature['label']
                                    .int64_list
                                    .value)
        print(label)


if __name__=='__main__':
    direct="./t/"
    tfrecord_filename="./4class_record_"
    tfRecord_cread(direct,tfrecord_filename)
    # tfRecord_test(tfrecord_filename)
