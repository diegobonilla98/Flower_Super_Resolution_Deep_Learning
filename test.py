from tensorflow.keras.models import load_model, Model
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from random import sample
import tensorflow as tf


def SubpixelConv2D(scale=2):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


def perceptual_loss(img_true, img_generated):
    img_true *= 127.5
    img_generated *= 127.5
    full_vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    loss_block3 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block3_conv3').output)
    loss_block3.trainable = False
    loss_block2 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
    loss_block2.trainable = False
    loss_block1 = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block1_conv2').output)
    loss_block1.trainable = False
    return K.mean(K.square(loss_block1(img_true) - loss_block1(img_generated))) + 2 * K.mean(
        K.square(loss_block2(img_true) - loss_block2(img_generated))) + 5 * K.mean(
        K.square(loss_block3(img_true) - loss_block3(img_generated)))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

full_vgg = vgg16.VGG16(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
full_vgg.trainable = False
for l in full_vgg.layers:
    l.trainable = False
model = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
model.trainable = False
def perceptual_loss(y_true, y_pred):
    y_pred_features = model(y_pred * 127.5)
    y_true_features = model(y_true * 127.5)
    return K.mean(K.square(y_true_features - y_pred_features))
sr = load_model('/media/bonilla/HDD_2TB_basura/models/FlowerSR/BonillaGAN/epoch_9750.h5', custom_objects={'SubpixelConv2D': SubpixelConv2D, 'perceptual_loss': perceptual_loss})
sr.summary()

data_loader = DataLoader('/media/bonilla/HDD_2TB_basura/databases/flowers/flower_photos/validation/roses/', (256, 256), 8)
x, y = data_loader.load_batch(3)

res = sr.predict(x)
# comb = (np.vstack([np.hstack([y[0], x[0], res[0]]), np.hstack([y[1], x[1], res[1]]), np.hstack([y[2], x[2], res[2]])]) + 1) / 2
comb = (np.vstack([np.hstack([x[0], res[0]]), np.hstack([x[1], res[1]]), np.hstack([x[2], res[2]])]) + 1) / 2

plt.figure(1, (7, 15))
plt.imshow(comb[:, :, ::-1])
plt.axis('off')
plt.show()
