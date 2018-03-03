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

    def get(self):
        if self.params:
            params = eval(self.params)
        else:
            params = {}
        return getattr(self, self.model)(**params)

    def acapellabot(self):
        mashup = Input(shape=(None, None, 1), name='input')
        convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(convA)
        conv = BatchNormalization()(conv)

        convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 4, strides=2, activation='relu',
                      padding='same', use_bias=False)(convB)
        conv = BatchNormalization()(conv)

        conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(128, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convB])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu',
                      padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convA])
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(1, 3, activation='relu', padding='same')(conv)
        acapella = conv
        return Model(inputs=mashup, outputs=acapella)

    def leaky_dropout(self, alpha1, alpha2, rate):
        mashup = Input(shape=(None, None, 1), name='input')
        dropout = Dropout(rate)(mashup)

        convA = Conv2D(64, 3, padding='same')(dropout)
        convA = LeakyReLU(alpha=alpha1)(convA)
        convA = Dropout(rate)(convA)

        conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(convA)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)

        convB = Conv2D(64, 3, padding='same')(conv)
        convB = LeakyReLU(alpha=alpha1)(convB)
        convB = Dropout(rate)(convB)

        conv = Conv2D(64, 4, strides=2, padding='same', use_bias=False)(convB)
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

        conv = Concatenate()([conv, convB])

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(64, 3, padding='same', use_bias=False)(conv)
        conv = LeakyReLU(alpha=alpha1)(conv)
        conv = Dropout(rate)(conv)

        conv = BatchNormalization()(conv)
        conv = UpSampling2D((2, 2))(conv)

        conv = Concatenate()([conv, convA])

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(64, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(32, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)
        conv = Dropout(rate)(conv)

        conv = Conv2D(1, 3, padding='same')(conv)
        conv = LeakyReLU(alpha=alpha2)(conv)

        acapella = conv
        return Model(inputs=mashup, outputs=acapella)
