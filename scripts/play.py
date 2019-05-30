# import tensorflow as tf
#
#
# def cond(i, result_list):
#     return tf.less(i, 10)
#
#
# def body(i, result_list):
#     i = tf.add(i, 1)
#     result_list.append(i)
#     return [i, result_list]
#
#
# result_list = []
# i = tf.constant(0)
# r = tf.while_loop(cond, body, [i, result_list])
#
# with tf.Session() as sess:
#     print(sess.run(r))

import tensorflow as tf
import numpy as np


def cond(step, output):
    return step < 10


def body(step, output):
    result = tf.range(5, dtype=tf.float32)
    # result = tf.cast(step, tf.float32) ** 2
    output = output.write(step, result)
    step = step + 1
    return step, output


step = tf.constant(0)
output = tf.TensorArray(dtype=tf.float32, size=10, dynamic_size=False)
final_step, final_output = tf.while_loop(cond, body, loop_vars=[step, output])
final_output = final_output.stack()
print(final_output)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(final_output))
