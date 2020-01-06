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
                 name,
                 target_shape=[-1, 28, 28, 1]):
    reshape_op = tf.reshape(x, target_shape)
    rotations_rad = rotation_rad(min_deg, max_deg, delta_deg)
    rots_cur = np.random.choice(rotations_rad, batch_sz)
    rot_op = tf.contrib.image.rotate(reshape_op, rots_cur)
    flatten_op = tf.reshape(rot_op, [-1, x.get_shape()[-1].value],
                            name=name+'/flatten_rotation')
    return flatten_op, rots_cur

def scale_ops(x,
                 min_scale, max_scale, delta_scale,
                 batch_sz,
                 name,
                 target_shape=[-1, 28, 28, 1]):
    reshape_op = tf.reshape(x, target_shape)
    _, height, width, _ = target_shape
    scales = np.arange(min_scale, max_scale+delta_scale, delta_scale)
    new_dims = [[max(1, s*height), max(1, s*width)] for s in scales]
    max_height, max_width = np.max(new_dims, axis=0)
    scales_idx_cur = np.random.choice(np.arange(len(new_dims)), batch_sz)
    scale_ops = [tf.image.resize_images(tf.gather(reshape_op, tf.constant(img_idx)), new_dims[sc_idx]) for img_idx, sc_idx in enumerate(scales_idx_cur)]
    padding = [[int(max_height-new_dims[sc_idx][0]/2), int(max_width-new_dims[sc_idx][1]/2)] for sc_idx in scales_idx_cur]
    pad_ops = [tf.pad(op, tf.constant([[ph,ph],[pw,pw]]), mode="REFLECT") for op, (ph, pw) in zip(scale_ops, padding)]
    flatten_op = tf.reshape(pad_ops, [-1, x.get_shape()[-1].value],
                            name=name+'/flatten_scale')
    return flatten_op, [new_dims[s_idx] for s_idx in scales_idx_cur]

def gaussian_noise_op(in_, std):
    prefix = in_.name.replace(':', '-')
    noise = tf.random_normal(shape=tf.shape(in_),
                             mean=0.0, stddev=std,
                             dtype=tf.float32,
                             name='_'.join([prefix, 'noise']))
    return tf.add(in_, noise, name='_'.join([prefix, 'noise_add']))
