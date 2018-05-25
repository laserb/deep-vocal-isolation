#!/usr/bin/python3
from keras.models import Model
from keras.layers import Input, Dropout, Conv2D, BatchNormalization, \
        UpSampling2D, Concatenate, LeakyReLU
from config import config


class Modeler(object):
    def __init__(self):
        self.config = config
        self.model = self.config.model
        self.params = self.config.model_params
        self.channels = self.config.get_channels()

    def get(self):
        if self.params:
            params = eval(self.params)
        else:
            params = {}
        return getattr(self, self.model)(**params)

    def acapellabot(self):
        mashup = Input(shape=(None, None, self.channels), name='input')
        conv_a = Conv2D(64, 3, activation='relu', padding='same')(mashup)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(conv_a)
        conv = BatchNormalization()(conv)

        conv_b = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(conv_b)
        conv = BatchNormalization()(conv)

        conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(128, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, conv_b])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, conv_a])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(self.channels,
                      3, activation='relu', padding='same')(conv)
        acapella = conv
        return Model(inputs=mashup, outputs=acapella)

    def leaky_dropout(self, alpha1, alpha2, rate):
        mashup = Input(shape=(None, None, self.channels), name='input')
        dropout = Dropout(rate)(mashup)

        conv_a = Conv2D(64, 3, padding='same')(dropout)
        conv_a = LeakyReLU(alpha=alpha1)(conv_a)
        conv_a = Dropout(rate)(conv_a)

        conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(conv_a)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)

        conv_b = Conv2D(64, 3, padding='same')(conv)
        conv_b = LeakyReLU(alpha=alpha1)(conv_b)
        conv_b = Dropout(rate)(conv_b)

        conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(conv_b)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)

        conv = Conv2D(128, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(128, 3, padding='same', use_bias=False)(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, conv_b])

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(64, 3, padding='same', use_bias=False)(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, conv_a])

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(32, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(self.channels, 3, padding='same')(conv)
        if not self.config.learn_phase:
            conv = LeakyReLU(alpha=alpha2)(conv)

        acapella = conv
        return Model(inputs=mashup, outputs=acapella)
