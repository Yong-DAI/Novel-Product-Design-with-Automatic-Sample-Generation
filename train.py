from keras.optimizers import Adam
from keras.utils import generic_utils
from ops import *
import keras.backend as K
import sys
import cv2

import numpy as np
from make_sketch import main_dy
# Utils
sys.path.append("utils/")
from simple_utils import plot_batch_train, plot_batch_eval, plot_batch_train_dy, plot_batch_eval_dy

import os
import time


def feature_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def pixel_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def variation_loss(y_true, y_pred):
    # Assume img size is 64*64
    if K.image_dim_ordering() == 'tf':
        a = K.square(y_pred[:, :64-1, :64-1, :] - y_pred[:, 1:, :64-1, :])
        b = K.square(y_pred[:, :64-1, :64-1, :] - y_pred[:, :64-1, 1:, :])
    else:
        a = K.square(y_pred[:, :, 64 - 1, :64 - 1] - y_pred[:, :, 1:, :64 - 1])
        b = K.square(y_pred[:, :, 64 - 1, :64 - 1] - y_pred[:, :, :64 - 1, 1:])
    return K.sum(K.sqrt(a+b))


def train(batch_size, nb_epoch, sketch, color, weights, tag, save_weight, img_dim):
#def train(batch_size, n_batch_per_epoch, nb_epoch, sketch, color, weights, tag, sk_ir, save_weight=1, img_dim=[64,64,1]):
    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model, model_name = edge2color(img_dim, batch_size=batch_size)

    model.compile(loss=[pixel_loss, feature_loss], loss_weights=[1, 1], optimizer=opt)
    model.summary()
    from keras.utils.visualize_util import plot
    plot(model, to_file='../figures/edge2color.png', show_shapes=True, show_layer_names=True)
    
    sketch = sketch
    color = color
    weights = weights

    global_counter = 1
    for epoch in range(nb_epoch):
        if epoch % 30 ==0 and epoch >1:
            print('System sleeps for 10 minutes')
            time.sleep(300)
#        
        batch_counter = 1
        start = time.time()
        
        batch_idxs = sketch.shape[0] // batch_size       #####  qu zheng chu   13136/16=821
        print (batch_idxs)
        n_batch_per_epoch = batch_idxs
#        n_batch_per_epoch = 25
        
        if n_batch_per_epoch >= batch_idxs or n_batch_per_epoch == 0:
            n_batch_per_epoch = batch_idxs
            
#        print (n_batch_per_epoch)

        progbar = generic_utils.Progbar(n_batch_per_epoch * batch_size)

########   dy
#        val_num = epoch%batch_idxs
#        sk_val = sketch[val_num* batch_size: (val_num + 1) * batch_size]
        
#        sk_val = sketch[0:16]
#        co_val = color[0:16]
#        sketch = sketch[16:13136]
#        color = color[16:13136]
#        weights = weights[16:13136]
        
#        pix_loss_sum =0
#        pix_loss_ave =0
#        fea_loss_sum =0
#        fea_loss_ave =0
        
        for idx in range(batch_idxs):
            
            batch_sk = sketch[idx * batch_size: (idx + 1) * batch_size]
            batch_co = color[idx * batch_size: (idx + 1) * batch_size]
            batch_weights = weights[idx * batch_size: (idx + 1) * batch_size]
            
            train_loss = model.train_on_batch([batch_sk], [batch_co, batch_weights])
            
            batch_counter += 1
            progbar.add(batch_size, values=[('pixel_loss', train_loss[1]), ('feature_loss', train_loss[2])])
#            pix_loss_sum +=train_loss[1]
#            fea_loss_sum +=train_loss[2]
            
#             if batch_counter >= n_batch_per_epoch:
#            ids_save_num = batch_idxs//5
#            if global_counter % ids_save_num == 0:               ####   50
            if global_counter % 50 == 0:               ####   50
                plot_batch_train_dy(model, img_dim[0], batch_size, batch_sk, batch_co, epoch, idx, tag,batch_idxs)
                
#                plot_batch_eval(model, img_dim[0], batch_size, sk_val, epoch,idx, batch_idxs, tag=tag+'_val')           ###  evalation
                
#                plot_batch_eval(model, img_dim[0], batch_size, sk_ir, tag=tag+'_test')
#                plot_batch_eval(model, img_dim[0], batch_size, co_val, tag=tag+'_test')
            global_counter += 1

            if batch_counter > n_batch_per_epoch:
                break
        
#        pix_loss_ave = pix_loss_sum/batch_counter
#        fea_loss_ave = fea_loss_sum/batch_counter
#        txt_screen = "Epoch: ({}) --pixel_loss: ({}) -- feature_loss: ({}) \n".format(epoch, pix_loss_ave, fea_loss_ave)
#        print (txt_screen)
        
        print ('Epoch %s/%s, Time: %s \n' % (epoch + 1, nb_epoch, time.time() - start))

        if save_weight:
            # save weights every epoch
            con_save_num = nb_epoch//20
            if (epoch+1) % con_save_num == 0 and epoch>0:
                weights_path = '%s/%s_weights_epoch_%s.h5' % (model_name, tag, epoch+1)
                if not os.path.exists('%s' % model_name):
                    os.mkdir('%s' % model_name)
                model.save_weights(weights_path, overwrite=True)


def evaluate(batch_size, tag, epoch, sketch, img_dim):
    model, model_name = edge2color(img_dim, batch_size=batch_size)
    model.load_weights('%s/%s_weights_epoch_%s.h5' % (model_name, tag, epoch))
    print ('Load Model Complete')
    plot_batch_eval_dy(model, img_dim[0], batch_size=batch_size, sketch=sketch, tag=tag, epoch=epoch)

def main():
    CUDA_VISIBLE_DEVICES= 0,1,2,3
    sketch, color, weights = load_with_size(os.path.expanduser
#                                   ('~/dy/doc_dy/keras-Convolutional_Sketch_Inversion-master/data/processed-3/lfw_96_data.h5'), img_size=96)
                                  ('../data/processed-3/lfw_200_data.h5'), img_size=200)

#    img_dim = [128, 128, 1]
    img_dim = [200, 200, 1]
    tag = '1-1-4batch-200'
    #####   n_batch_per_epoch=4,
    
    train(batch_size=16,  nb_epoch=500, sketch=sketch, color=color, weights=weights, tag=tag, save_weight=1, img_dim=img_dim)

def main2():
    sketch = main_dy().astype(np.float32)
    img_size=200
    sketch = sketch.transpose((2, 3, 0, 1))    #####[0,0,64,64]
    sketch = sketch.reshape( img_size, img_size,1)
    test_sketch = np.ndarray((16,img_size, img_size,1))
    for i in range(16):
        test_sketch[i] = sketch
    
#    filename = "../figures/sketch_dy/sketch-0.jpg"
#    sketch = cv2.imread(filename)
    img_dim = [200, 200, 1]
    tag = '1-1-4batch-200'
    evaluate(batch_size=16, tag=tag, epoch = 1400, sketch=test_sketch, img_dim= img_dim)


if __name__ == '__main__':
    main()
