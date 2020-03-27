import tensorflow as tf

i = tf.constant(0)
while_condition = lambda i: tf.less(i, tf.placeholder(tf.int32, [None,5]))
def body(i):
    # do something here which you want to do in your loop
    # increment i
    return [tf.add(i, 1)]

# do the loop:
r = tf.while_loop(while_condition, body, [i])