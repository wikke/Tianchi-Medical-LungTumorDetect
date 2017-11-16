from keras.models import Model
from keras.layers import Input, Conv3D, Dense, BatchNormalization, Concatenate, AveragePooling3D, GlobalAveragePooling3D, Activation
from keras.optimizers import Adam
from config import *

def get_DenseNet_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = Conv3D(DENSE_NET_INITIAL_CONV_DIM, (3, 3, 3), padding='same')(inputs)

    for i in range(DENSE_NET_BLOCKS):
        y = BatchNormalization()(x)
        y = Activation('relu')(y)
        y = Conv3D(DENSE_NET_DIM_INCR, (3, 3, 3), padding='same')(y)
        x = Concatenate(axis=4)([x, y])

        if (i + 1) % DENSE_NET_SHRINKAGE_STEPS == 0 and i != DENSE_NET_BLOCKS - 1:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv3D(x.get_shape()[4].value, (3, 3, 3), padding='same')(x)
            x = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model
