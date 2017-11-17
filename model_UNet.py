from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K
from config import *
from generators import get_seg_batch
from skimage import morphology, measure, segmentation
from visual_utils import plot_slices, plot_comparison
import numpy as np

SMOOTH = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def metrics_true_sum(y_true, y_pred):
    return K.sum(y_true)

def metrics_pred_sum(y_true, y_pred):
    return K.sum(y_pred)

def metrics_pred_max(y_true, y_pred):
    return K.max(y_pred)

def metrics_pred_min(y_true, y_pred):
    return K.min(y_pred)

def metrics_pred_mean(y_true, y_pred):
    return K.mean(y_pred)

def do_evaluate(model):
    print('Model evaluating')
    X, y_true = next(get_seg_batch(1, from_train=False, random_choice=True))
    y_pred = model.predict(X)

    X, y_true, y_pred = X[0,:,:,:,0], y_true[0,:,:,:,0], y_pred[0,:,:,:,0]
    intersection = y_true * y_pred
    recall = (np.sum(intersection) + SMOOTH) / (np.sum(y_true) + SMOOTH)
    precision = (np.sum(intersection) + SMOOTH) / (np.sum(y_pred) + SMOOTH)
    print('Average recall {:.4f}, precision {:.4f}'.format(recall, precision))

    for threshold in range(0, 10, 2):
        threshold = threshold / 10.0
        pred_mask = (y_pred > threshold).astype(np.uint8)
        intersection = y_true * pred_mask
        recall = (np.sum(intersection) + SMOOTH) / (np.sum(y_true) + SMOOTH)
        precision = (np.sum(intersection) + SMOOTH) / (np.sum(y_pred) + SMOOTH)
        print("Threshold {}: recall {:.4f}, precision {:.4f}".format(threshold, recall, precision))

    regions = measure.regionprops(measure.label(y_pred))
    print('Num of pred regions {}'.format(len(regions)))

    if DEBUG_PLOT_WHEN_EVALUATING_SEG:
        plot_comparison(X, y_true, y_pred)
        plot_slices(X)
        plot_slices(y_true)
        plot_slices(y_pred)

class UNetEvaluator(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        if self.counter % TRAIN_SEG_EVALUATE_FREQ == 0:
            do_evaluate(self.model)

def get_unet():
    return get_simplified_unet() if USE_SIMPLIFIED_UNET else get_full_unet()

def get_simplified_unet():
    inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    # pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    # up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=-1)
    # conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    # conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    # up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=dice_coef_loss,
                  metrics=[dice_coef, metrics_true_sum, metrics_pred_sum, metrics_pred_max, metrics_pred_min, metrics_pred_mean])

    return model

def get_full_unet():
    inputs = Input((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(OUTPUT_CHANNEL, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=dice_coef_loss,
                  metrics=[dice_coef, metrics_true_sum, metrics_pred_sum, metrics_pred_max, metrics_pred_min, metrics_pred_mean])

    return model

if __name__ == '__main__':
    model = get_unet()
    model.summary()
