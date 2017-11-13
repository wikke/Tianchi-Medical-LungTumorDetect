from glob import glob
import numpy as np
import pandas as pd
import h5py
from random import randint
import _pickle as pickle
from config import *

def get_tumor_records():
    img_numpy_files = glob('{}/*.h5'.format(PREPROCESS_PATH))
    meta_data = {}
    for f in glob('{}/*.meta'.format(PREPROCESS_PATH)):
        with open(f, 'rb') as f:
            meta = pickle.load(f)
            meta_data[meta['seriesuid']] = meta

    def find_numpy_file(seriesuid):
        for f in img_numpy_files:
            if f[-13:-3] == seriesuid:
                return f
        return None
    def find_origin(seriesuid):
        return meta_data[seriesuid]['origin'] if seriesuid in meta_data else None
    def find_spacing(seriesuid):
        return meta_data[seriesuid]['spacing'] if seriesuid in meta_data else None

    records = pd.read_csv('{}/train/annotations.csv'.format(ANNOTATIONS_PATH))
    records['img_numpy_file'] = records['seriesuid'].apply(find_numpy_file)
    records['origin'] = records['seriesuid'].apply(find_origin)
    records['spacing'] = records['seriesuid'].apply(find_spacing)

    records.dropna(inplace=True)

    return records

tumor_records = get_tumor_records()
if tumor_records.shape[0] == 0:
    print('no tumor records, generator cannot work')
    exit()

def get_image_and_records(seriesuid):
    records = tumor_records[tumor_records['seriesuid'] == seriesuid]
    if records.shape[0] == 0:
        print('eva ct, no records of seriesuid {}'.format(seriesuid))
        return None, None

    img = None
    with h5py.File(records.iloc[0]['img_numpy_file'], 'r') as hf:
        img = np.zeros(hf['img'].shape)
        img[:] = hf['img'][:]

    return img / DEBUG_IMAGE_STD, records

def get_block(record, around_tumor=True):
    with h5py.File(record['img_numpy_file'], 'r') as hf:
        H, W, D = hf['img'].shape[0], hf['img'].shape[1], hf['img'].shape[2]

        if around_tumor:
            coord = np.array([record['coordX'], record['coordY'], record['coordZ']])
            coord = np.abs((coord - record['origin']) / record['spacing'])
            h, w, d = int(coord[0] - INPUT_HEIGHT // 2), int(coord[1] - INPUT_WIDTH // 2), int(coord[2] - INPUT_DEPTH // 2)

            h, w, d = max(h, 0), max(w, 0), max(d, 0)
            h, w, d = min(h, H - INPUT_HEIGHT - 1), min(w, W - INPUT_WIDTH - 1), min(d, D - INPUT_DEPTH - 1)
        else:
            h, w, d = randint(0, H - INPUT_HEIGHT - 1), randint(0, W - INPUT_WIDTH - 1), randint(0, D - INPUT_DEPTH - 1)

        return hf['img'][h:h + INPUT_HEIGHT, w:w + INPUT_WIDTH, d:d + INPUT_DEPTH] / DEBUG_IMAGE_STD

# TODO enchanced? rotate, random.
def get_seg_batch(batch_size=32):
    idx = 0
    X = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))
    y = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, OUTPUT_CHANNEL))

    while True:
        for b in range(batch_size):
            record = tumor_records.iloc[idx]
            # print('get batch idx {}, seriesuid {}'.format(idx, record['seriesuid']))

            X[b,:,:,:,0] = get_block(record, around_tumor=True)
            y[b,:,:,:,0] = make_seg_mask(record)

            if not DEBUG_SEG_TRY_OVERFIT:
                idx = idx + 1 if idx < tumor_records.shape[0] - 1 else 0

        yield X, y

def make_seg_mask(record):
    mask = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, INPUT_DEPTH))
    radius = int(record['diameter_mm'] // 2 + DIAMETER_BUFFER)

    mask[INPUT_HEIGHT//2-radius:INPUT_HEIGHT//2+radius,
         INPUT_WIDTH//2-radius:INPUT_WIDTH//2+radius,
         INPUT_DEPTH//2-radius:INPUT_DEPTH//2+radius] = 1.0

    return mask

# TODO enchanced? rotate???
# [1, 0], positive sample
# [0, 1], negative sample
def get_classify_batch(batch_size=32):
    idx = 0
    X = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))
    y = np.zeros((batch_size, 2))
    positive_num = int(batch_size * CLASSIFY_POSITIVE_SAMPLE_RATIO)

    while True:
        for b in range(positive_num):
            record = tumor_records.iloc[idx]
            X[b,:,:,:,0] = get_block(record, around_tumor=True)
            y[b,0] = 1

            idx = idx + 1 if idx < tumor_records.shape[0] - 1 else 0

        for b in range(positive_num, batch_size):
            record = tumor_records.iloc[randint(0, tumor_records.shape[0] - 1)]
            X[b,:,:,:,0] = get_block(record, around_tumor=False)
            y[b,1] = 1

        yield X, y
