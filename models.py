from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
# from collections import deque
import config
import sys

class TestModels():
    def __init__(self, nclasses, model, seq_length=config.MIN_SEQ_LENGTH, saved_model=None):
        self.nclasses = nclasses
        self.saved_model = saved_model

        if self.saved_model is not None:
            self.model = load_model(self.saved_model)
        elif model == 'lrcn':
            self.input_shape = (None, config.IMG_WIDTH, config.IMG_HEIGHT, config.IMG_CHANNELS)
            self.model = self.lrcn()
        else:
            print("No such network configuration: {0}" % model)
            sys.exit()
        
        metrics = ['accuracy']
        if self.nclasses >= 10:
            metrics.append('top_k_categorical_accuracy')

        optimizer = Adam(lr=1e-5, decay=1e-6)
        if self.nclasses>2:
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        else:
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)


    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py
        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556
        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(64, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(64, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(128, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(128, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # model.add(TimeDistributed(Conv2D(256, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(256, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
        
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(Conv2D(512, (3,3),
        #     padding='same', activation='relu')))
        # model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(32, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nclasses, activation='softmax'))

        return model