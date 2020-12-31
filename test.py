from tensorflow.keras.models import load_model, Model
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from random import sample
import tensorflow as tf


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
sr = load_model('/media/bonilla/HDD_2TB_basura/models/FlowerSR/autoencoder_model.h5', custom_objects={'perceptual_loss': perceptual_loss, 'InstanceNormalization': InstanceNormalization})


data_loader = DataLoader('/media/bonilla/HDD_2TB_basura/databases/flowers/flower_photos/validation/daisy/', (256, 256), 8)
x, y = data_loader.load_batch(3)

res = sr.predict(x)
# comb = (np.vstack([np.hstack([y[0], x[0], res[0]]), np.hstack([y[1], x[1], res[1]]), np.hstack([y[2], x[2], res[2]])]) + 1) / 2
comb = (np.vstack([np.hstack([x[0], res[0]]), np.hstack([x[1], res[1]]), np.hstack([x[2], res[2]])]) + 1) / 2

plt.figure(1, (7, 7))
plt.imshow(comb[:, :, ::-1])
plt.axis('off')
plt.show()
