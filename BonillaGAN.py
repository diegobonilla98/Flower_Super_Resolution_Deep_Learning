from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, LeakyReLU, PReLU, add, Dropout, BatchNormalization, Lambda, Activation, Dense, Flatten
from InstanceNormalization import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


full_vgg = vgg16.VGG16(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
full_vgg.trainable = False
for l in full_vgg.layers:
    l.trainable = False
features_model = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
features_model.compile(loss='mse', optimizer=Adam(1e-4, 0.9))
features_model.trainable = False


def preprocess_vgg(x):
    if isinstance(x, np.ndarray):
        return vgg16.preprocess_input((x + 1) * 127.5)
    else:
        return Lambda(lambda x: vgg16.preprocess_input(tf.add(x, 1) * 127.5))(x)

    # def perceptual_loss(y_true, y_pred):
#     y_pred_features = features_model(y_pred * 127.5)
#     y_true_features = features_model(y_true * 127.5)
#     return K.mean(K.square(y_true_features - y_pred_features))


class GANSR:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.gf = 128
        self.channels = 3
        self.image_shape = (256, 256, 3)

        self.generator = self.build_generator()
        self.generator.compile(Adam(1e-5, 0.9), 'mse')
        self.generator.summary()
        plot_model(self.generator, to_file='generator_model.png')

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, 0.9))
        self.discriminator.summary()
        plot_model(self.discriminator, to_file='discriminator_model.png')

        self.discriminator.trainable = False
        input_tensor = Input(shape=self.image_shape)
        gen = self.generator(input_tensor)
        generated = self.discriminator(gen)
        features = features_model(preprocess_vgg(gen))
        self.adversarial = Model(input_tensor, [generated, features])
        self.adversarial.summary()
        self.adversarial.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[1e-3, 0.006], optimizer=Adam(1e-5, 0.9))
        plot_model(self.adversarial, to_file='adversarial_model.png')

    def build_discriminator(self):
        filters = 64
        input_tensor = Input(shape=self.image_shape)

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        x = conv2d_block(input_tensor, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(input_tensor, output)

        return model

    @staticmethod
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

    def build_generator(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name)(layer_input)
            d = BatchNormalization(name=name + "_bn")(d)
            d = PReLU(shared_axes=[1, 2])(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2")(d)
            d = BatchNormalization(name=name + "_bn2")(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
            u = self.SubpixelConv2D()(u)
            u = BatchNormalization(name=name + "_bn")(u)
            u = PReLU(shared_axes=[1, 2])(u)
            return u

        # Image input
        input_tensor = Input(shape=self.image_shape)
        c1 = conv2d(input_tensor, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=3, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=3, strides=2, name='g_d2_dc')

        output_img = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(d2)

        return Model(inputs=input_tensor, outputs=output_img)

    def plot_images(self, epoch):
        x, y = self.data_loader.load_batch(batch_size=3)
        res = self.generator.predict(x)
        comb = (np.vstack([np.hstack([y[0], x[0], res[0]]), np.hstack([y[1], x[1], res[1]]), np.hstack([y[2], x[2], res[2]])]) + 1) / 2
        plt.imshow(comb[:, :, ::-1])
        plt.axis('off')
        plt.savefig(f'./results/epoch_{epoch}.jpg')

    def fit(self, epochs, batch_size):
        for epoch in range(epochs):
            real_X, real_Y = self.data_loader.load_batch(batch_size=batch_size)
            real_y = np.ones((batch_size, 16, 16, 1)) * 0.9
            # real_y = np.random.uniform(0.75, 1., size=(batch_size, 6, 6, 1))

            fake_X = self.generator.predict(real_X)
            fake_y = np.zeros((batch_size, 16, 16, 1))
            # fake_y = np.random.uniform(0., 0.3, size=(batch_size, 6, 6, 1))

            d_loss_true = self.discriminator.train_on_batch(real_Y, real_y)
            d_loss_fake = self.discriminator.train_on_batch(fake_X, fake_y)

            # real_y = np.random.uniform(0.75, 1., size=(batch_size, 6, 6, 1))
            feat = features_model.predict(preprocess_vgg(real_Y))
            g_loss = self.adversarial.train_on_batch(real_X, [real_y, feat])

            print(f"[Epoch: {epoch}/{epochs}]\t[adv_loss: {g_loss}, d_fake: {d_loss_fake}, d_true: {d_loss_true}]")

            if epoch % 25 == 0:
                self.plot_images(epoch)


data_loader = DataLoader('/media/bonilla/HDD_2TB_basura/databases/all_flowers/', (256, 256), 8)
asr = GANSR(data_loader)
asr.fit(10_000, 4)
