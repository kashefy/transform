'''
Created on Nov 4, 2017

@author: kashefy
'''
import numpy as np
import tensorflow as tf

def rotation_rad(min_deg, max_deg, delta_deg):
    return np.deg2rad(np.arange(min_deg, max_deg+delta_deg, delta_deg)).tolist()

def rotation_ops(x, 
                 min_deg, max_deg, delta_deg,
                 batch_sz,
                 name):
    reshape_op = tf.reshape(x, [-1, 28, 28, 1])
    rotations_rad = rotation_rad(min_deg, max_deg, delta_deg)
    rots_cur = np.random.choice(rotations_rad, batch_sz)
    rot_op = tf.contrib.image.rotate(reshape_op, rots_cur)
    flatten_op = tf.reshape(rot_op, [-1, x.get_shape()[-1].value],
                            name=name+'/flatten_rot')
    return flatten_op, rots_cur

def gaussian_noise_op(in_, std):
    prefix = in_.name.replace(':', '-')
    noise = tf.random_normal(shape=tf.shape(in_),
                             mean=0.0, stddev=std,
                             dtype=tf.float32,
                             name='_'.join([prefix, 'noise']))
    return tf.add(in_, noise, name='_'.join([prefix, 'noise_add']))
