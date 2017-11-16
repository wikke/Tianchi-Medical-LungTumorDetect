from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization
from keras.optimizers import Adam
from config import *

def get_simplified_VGG_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))

    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(inputs)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    # x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    #
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    # x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling3D()(x)
    # x = Flatten()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def get_full_VGG_classifier():
    inputs = Input((CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    x = inputs


    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling3D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=TRAIN_CLASSIFY_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    return model

