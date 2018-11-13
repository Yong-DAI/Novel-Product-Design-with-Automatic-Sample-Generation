
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Lambda
from keras.layers import Input, merge
from keras.regularizers import l2
from keras.models import Model
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import h5py


def load_with_size(db_name, img_size):
    with h5py.File(db_name, 'r') as hf:
        sketch = np.array(hf['lfw_%s_sketch' % img_size]).astype(np.float32)
        sketch = sketch.transpose((0, 2, 3, 1))
#        color = np.array(hf['lfw_%s_img' % img_size]).astype(np.float32) / 255.
        color = np.array(hf['lfw_%s_color' % img_size]).astype(np.float32) / 255.
        color = color.transpose((0, 2, 3, 1))
        weights = np.array(hf['lfw_%s_vgg' % img_size])
        
#        sk_ir = np.array(hf['ir_%s_sketch' % img_size])[0:16]
#        sk_ir = np.array(hf['lfw_%s_sketch' % img_size])[0:16]
#        sk_ir = sk_ir.transpose((0, 2, 3, 1))
        print ('sketch data has shape:', sketch.shape)
        print ('color data has shape:', color.shape)
        print ('vgg_16 weights data has shape', weights.shape)
#        print ('IR sketch data sample has shape', sk_ir.shape)
    return sketch, color, weights



def load(db_name):
    with h5py.File(db_name, 'r') as hf:
        sketch = np.array(hf['lfw_sketch_data']).astype(np.float32)
        sketch = sketch.transpose((0, 2, 3, 1))
        color = np.array(hf['lfw_64_data']).astype(np.float32) / 255.
        color = color.transpose((0, 2, 3, 1))
        weights = np.array(hf['vgg_16_weight_relu22'])
        print ('sketch data has shape:', sketch.shape)
        print ('color data has shape:', color.shape)
        print ('vgg_16 weights data has shape', weights.shape)
        return sketch, color, weights

def load_ir_sk_with_size(db_name, img_size):
    with h5py.File(db_name,'r') as hf:
        sketch = np.array(hf['ir_%s_sketch' % img_size]).astype(np.float32)
        sketch = sketch.transpose((0, 2, 3, 1))
        print ('IR sketch data has shape:', sketch.shape)
    return sketch

def residual_block(x, block_idx, nb_filter, bn=True, weight_decay=0, k_size=3):

    # 1st conv
    name = "block%s_conv2D%s" % (block_idx, "a")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, k_size, k_size, border_mode="same", W_regularizer=W_reg, name=name)(x)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "a"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "a"))(r)

    # 2nd conv
    name = "block%s_conv2D%s" % (block_idx, "b")
    W_reg = l2(weight_decay)
    r = Convolution2D(nb_filter, k_size, k_size, border_mode="same", W_regularizer=W_reg, name=name)(r)
    if bn:
        r = BatchNormalization(mode=2, axis=1, name="block%s_bn%s" % (block_idx, "b"))(r)
    r = Activation("relu", name="block%s_relu%s" % (block_idx, "b"))(r)

    # Merge residual and identity
    x = merge([x, r], mode='sum', concat_axis=1, name="block%s_merge" % block_idx)

    return x


def convolutional_block(x, block_idx, nb_filter, k_size=3, subsample=(1, 1)):
    name = "block%s_conv2D" % block_idx
    x = Convolution2D(nb_filter, k_size, k_size, name=name, border_mode="same", subsample=subsample)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    return x


def rescale(x):
    x = (x + 1) / 2.
    return x


def last_convolutional_block(x, block_idx, nb_filter, k_size=3, subsample=(1, 1)):
    name = "block%s_conv2D" % block_idx
    x = Convolution2D(nb_filter, k_size, k_size, name=name, border_mode="same", subsample=subsample)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("sigmoid")(x)
    # # scale to (0,1) if use tanh
    # rescale_layer = Lambda(rescale)
    # x = rescale_layer(x)
    return x


def deconvolutional_block(x, block_idx, nb_filter, output_shape, k_size=3, subsample=(2, 2)):
    name = "block%s_deconv2D" % block_idx
    x = Deconvolution2D(nb_filter, k_size, k_size, output_shape=output_shape, name=name, border_mode='same', subsample=subsample)(x)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Activation("relu")(x)
    return x


def preprocess_VGG(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    # x has pixels intensities between 0 and 1
    x = 255. * x
    norm_vec = K.variable([103.939, 116.779, 123.68])
    if dim_ordering == 'th':
        norm_vec = K.reshape(norm_vec, (1,3,1,1))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        norm_vec = K.reshape(norm_vec, (1,1,1,3))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def edge2color(img_dim, batch_size):
    x_input = Input(shape=img_dim, name='input')
    # first 3 conv blocks
    h1 = convolutional_block(x_input, 1, 32, k_size=9, subsample=(1, 1))
    h2 = convolutional_block(h1, 2, 64, k_size=3, subsample=(2, 2))
    h3 = convolutional_block(h2, 3, 128, k_size=3, subsample=(2, 2))
    # then 5 res blocks
    h4 = residual_block(h3, 4, 128, k_size=3)
    h5 = residual_block(h4, 5, 128, k_size=3)
    h6 = residual_block(h5, 6, 128, k_size=3)
    h7 = residual_block(h6, 7, 128, k_size=3)
    h8 = residual_block(h7, 8, 128, k_size=3)
    # the 2 deconv blocks
    h9 = deconvolutional_block(h8, 9, 64, k_size=3,
                               output_shape=(batch_size, int(img_dim[0]/2), int(img_dim[1]/2), 64), subsample=(2, 2))
    h10 = deconvolutional_block(h9, 10, 32, k_size=3,
                                output_shape=(batch_size, img_dim[0], img_dim[1], 32), subsample=(2, 2))
    # final conv block
    h11 = last_convolutional_block(h10, 11, 3, k_size=9, subsample=(1, 1))

    # extract vgg feature
    vgg_16 = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None)
    # freeze VGG_16 when training
    for layer in vgg_16.layers:
        layer.trainable = False
    vgg_first2 = Model(input=vgg_16.input, output=vgg_16.get_layer('block2_conv2').output)
    Norm_layer = Lambda(preprocess_VGG)
    x_VGG = Norm_layer(h11)
    feat = vgg_first2(x_VGG)

    # build model
    model = Model(input=x_input, output=[h11, feat], name='edge2color')
    return model, model.name
