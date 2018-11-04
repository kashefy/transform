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
    
def scaled_dims(height, width, min_scale, max_scale, delta_scale):
    scales = np.arange(min_scale, max_scale+delta_scale, delta_scale)
    new_dims = [[max(1, int(s*height)), max(1, int(s*width))] for s in scales]
    return new_dims

def scale_translate_ops(x,
                 new_dims,
                 batch_sz,
                 name,
                 target_shape=[-1, 28, 28, 1]):
    reshape_op = tf.reshape(x, target_shape)
    _, height, width, num_channels = target_shape
    max_height, max_width = np.max(new_dims, axis=0)
    scales_idx_cur = np.random.choice(len(new_dims), batch_sz, replace=True)
    scale_ops = [tf.image.resize_images(tf.gather(reshape_op, tf.constant(img_idx)), new_dims[sc_idx]) for img_idx, sc_idx in enumerate(scales_idx_cur)]
    pad_ops, padding = pad_to_ops(scale_ops, max_height, max_width, [new_dims[sc_idx] for sc_idx in scales_idx_cur])
    flatten_op = tf.reshape(pad_ops, [-1, max_height*max_width*num_channels],
                            name=name+'/flatten_scale')
    bbox_top_left = [[ph[0], pw[0]] for ph, pw in padding]
    return flatten_op, [new_dims[s_idx] for s_idx in scales_idx_cur], bbox_top_left
    
def pad_to_ops(in_list, target_height, target_width, cur_dims):
    dim_diff = [[target_height-cur_h, target_width-cur_w] for cur_h, cur_w in cur_dims]
    padding = [[[int(np.floor(dh/2.)), int(np.ceil(dh/2.))], [int(np.floor(dw/2.)), int(np.ceil(dw/2.))]] for dh, dw in dim_diff]
    for idx, (ph, pw) in enumerate(padding):
        if ph[0] != ph[1] or pw[0] != pw[1]:
            flip_mode = np.random.randint(4)
            if flip_mode == 2: # vertical
                padding[idx][0] = padding[idx][0][::-1]
            elif flip_mode == 1: # horizontal
                padding[idx][1] = padding[idx][1][::-1]
            elif flip_mode == 2: # both
                padding[idx][0] = padding[idx][0][::-1]
                padding[idx][1] = padding[idx][1][::-1]
    pad_ops = [tf.pad(op, tf.constant([ph, pw, [0,0]]), mode="CONSTANT", constant_values=0) for op, (ph, pw) in zip(in_list, padding)]
    return pad_ops, padding

def gaussian_noise_op(in_, std):
    prefix = in_.name.replace(':', '-')
    noise = tf.random_normal(shape=tf.shape(in_),
                             mean=0.0, stddev=std,
                             dtype=tf.float32,
                             name='_'.join([prefix, 'noise']))
    return tf.add(in_, noise, name='_'.join([prefix, 'noise_add']))
