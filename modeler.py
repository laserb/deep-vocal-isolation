#!/usr/bin/python3
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, \
        UpSampling2D, Concatenate
from config import Config


class Modeler(object):
    def __init__(self):
        self.config = Config()
        self.model = self.config.model

    def get(self):
        return getattr(self, self.model)()

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
