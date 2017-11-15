from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K

from config import *
from generators import get_seg_batch, get_image_and_records
from skimage import morphology, measure, segmentation
from visual_utils import plot_slices, plot_comparison
import numpy as np

SMOOTH = 0.0001

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def evaluate_ct(model, seriesuid):
    print('************ Evaluating ************')
    X_whole, tumor_records = get_image_and_records(seriesuid)
    if X_whole is None:
        print('no records')
        return

    if DEBUG_PART_EVALUATE_CALLBACK:
        record = tumor_records.iloc[0]
        coord = np.array([record['coordX'], record['coordY'], record['coordZ']])
        coord = np.abs((coord - record['origin']) / record['spacing'])
        coord = coord - np.array([INPUT_WIDTH // 2, INPUT_HEIGHT // 2, INPUT_DEPTH // 2])
        coord = coord.astype(np.uint16)
        X_whole = X_whole[coord[0]:coord[0]+INPUT_WIDTH, coord[1]:coord[1]+INPUT_HEIGHT, coord[2]:coord[2]+INPUT_DEPTH]
        print('PART_EVALUATE DEBUG ENABLED: X_whole shape {}'.format(X_whole.shape))

    y_whole = np.zeros(X_whole.shape)
    for i in range(tumor_records.shape[0]):
        record = tumor_records.iloc[i]
        coord = np.array([record['coordX'], record['coordY'], record['coordZ']])
        coord = np.abs((coord - record['origin']) / record['spacing']).astype(np.uint16)

        if DEBUG_PART_EVALUATE_CALLBACK:
            coord = np.array([INPUT_WIDTH//2, INPUT_HEIGHT//2, INPUT_DEPTH//2])

        r = record['diameter_mm'] / 2 + DIAMETER_BUFFER
        radius = np.array([r, r, r])

        if DIAMETER_SPACING_EXPAND:
            radius = radius / record['spacing']

        radius = radius.astype(np.uint16)

        y_whole[coord[0] - radius[0]:coord[0] + radius[0] + 1,
                coord[1] - radius[1]:coord[1] + radius[1] + 1,
                coord[2] - radius[2]:coord[2] + radius[2] + 1] = 1.0

    pred_whole = np.zeros(X_whole.shape)
    for w in range(0, X_whole.shape[0] - INPUT_WIDTH + 1, INPUT_WIDTH):
        for h in range(0, X_whole.shape[1] - INPUT_HEIGHT + 1, INPUT_HEIGHT):
            for d in range(0, X_whole.shape[2] - INPUT_DEPTH + 1, INPUT_DEPTH):
                # print('in {}'.format((h,w,d)))
                _x = np.expand_dims(np.expand_dims(X_whole[w:w+INPUT_WIDTH, h:h+INPUT_HEIGHT, d:d+INPUT_DEPTH], axis=-1), axis=0)
                _pred = model.predict(_x)
                pred_whole[w:w + INPUT_WIDTH, h:h + INPUT_HEIGHT, d:d + INPUT_DEPTH] = _pred[0,:,:,:,0]

    intersection_whole = y_whole * pred_whole
    dice_coef = (2. * np.sum(intersection_whole) + SMOOTH) / (np.sum(y_whole) + np.sum(pred_whole) + SMOOTH)
    print("ceof {:.4f}, loss {:.4f}".format(dice_coef, 1 - dice_coef))

    y_sum, y_count = np.sum(y_whole), np.count_nonzero(y_whole)
    pred_sum, pred_count = np.sum(pred_whole), np.count_nonzero(pred_whole)
    inter_sum, inter_count = np.sum(intersection_whole), np.count_nonzero(intersection_whole)
    print('y y_sum {:.4f}, y_count {}'.format(y_sum, y_count))
    print('pred_sum {:.4f}, pred_count {}'.format(pred_sum, pred_count))
    print('inter_sum {:.4f}, inter_count {}'.format(inter_sum, inter_count))

    for threshold in range(0, 10, 2):
        threshold = threshold / 10.0
        pred_thres = (pred_whole > threshold).astype(np.uint8)
        inter_thres = y_whole * pred_thres
        dice_coef = (2. * np.sum(inter_thres) + SMOOTH) / (np.sum(y_whole) + np.sum(pred_thres) + SMOOTH)
        recall = np.sum(inter_thres) / np.sum(y_whole)
        precision = np.sum(inter_thres) / np.sum(pred_thres)
        print("threshold {}: ceof {:.4f}, loss {:.4f}, recall {:.4f}, precision {:.4f}".format(threshold, dice_coef, 1 - dice_coef, recall, precision))

    regions = measure.regionprops(measure.label(pred_whole))
    print('Num of pred regions {}'.format(len(regions)))

    # for region in regions:
    #     properties = {
    #         'area': region.area,
    #         'bbox': region.bbox,
    #         'centroid': region.centroid,
    #         # 'coords': region.coords,
    #         'equivalent_diameter': region.equivalent_diameter,
    #         'extent': region.extent,
    #         'filled_area': region.filled_area,
    #         'label': region.label,
    #     }
    #     print(properties)

    # plot_comparison(X_whole, y_whole, pred_whole)

class UNetEvaluator(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        if self.counter % TRAIN_SEG_EVALUATE_FREQ == 0:
            evaluate_ct(self.model, 'LKDS-00052')

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
    model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=dice_coef_loss, metrics=[dice_coef])

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
    model.compile(optimizer=Adam(lr=TRAIN_SEG_LEARNING_RATE), loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == '__main__':
    model = get_unet()
    model.summary()
