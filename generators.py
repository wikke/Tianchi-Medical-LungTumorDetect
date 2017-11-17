import os
from glob import glob
import numpy as np
import pandas as pd
import h5py
import random
import _pickle as pickle
from config import *
from visual_utils import plot_middle_slices_comparison

def get_meta_dict():
    cache_file = '{}/all_meta_cache.meta'.format(PREPROCESS_PATH)
    if os.path.exists(cache_file):
        print('get meta_dict from cache')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    meta_dict = {}
    for f in glob('{}/*.meta'.format(PREPROCESS_PATH)):
        seriesuid = f[-15:-5]
        if not os.path.exists('{}/{}.h5'.format(PREPROCESS_PATH, seriesuid)):
            continue

        with open(f, 'rb') as f:
            meta = pickle.load(f)
            meta_dict[meta['seriesuid']] = meta

    # cache it
    with open(cache_file, 'wb') as f:
        pickle.dump(meta_dict, f)

    return meta_dict

def get_tumor_records():
    numpy_files = glob('{}/*.h5'.format(PREPROCESS_PATH))
    meta_dict = get_meta_dict()

    fields = ['img_numpy_file', 'origin', 'spacing', 'shape', 'pixels', 'cover_ratio', 'process_duration']
    def fill_info(seriesuid):
        data = [None] * len(fields)

        for f in numpy_files:
            if f[-13:-3] == seriesuid:
                data[0] = f

        if seriesuid in meta_dict:
            t = meta_dict[seriesuid]
            data[1:] = [t['origin'], t['spacing'], t['shape'], t['pixels'], t['cover_ratio'], t['process_duration']]

        return pd.Series(data, index=fields)

    records = pd.read_csv('{}/train/annotations.csv'.format(ANNOTATIONS_PATH))
    records[fields] = records['seriesuid'].apply(fill_info)
    records.dropna(inplace=True)

    print('tumor record size {}'.format(records.shape))
    if DEBUG_ONLY_TRAIN_FINE_CUT_BIG_TUMOR_SWITCHER:
        records.drop(records[records.cover_ratio < DEBUG_ONLY_TRAIN_COVER_RATIO_BIGGER_THAN].index, axis=0, inplace=True)
        records.drop(records[records.diameter_mm < DEBUG_ONLY_TRAIN_TUMOR_DIAMETER_LARGER_THAN].index, axis=0, inplace=True)
        print('after drop, tumor record size {}'.format(records.shape))

    return records

tumor_records = get_tumor_records()
tumor_records_len = tumor_records.shape[0]
if tumor_records_len == 0:
    print('no tumor records, generator cannot work')
    exit()

if RANDOMIZE_RECORDS:
    tumor_records = tumor_records.iloc[np.random.permutation(tumor_records_len)]
    print('tumor_records_train randomized')

tumor_records_train = tumor_records[:int(tumor_records_len * TRAIN_VAL_RATIO)]
tumor_records_val = tumor_records[int(tumor_records_len * TRAIN_VAL_RATIO):]
del tumor_records, tumor_records_len

re_sample = (TRAIN_CLASSIFY_NOT_SEGMENTATION and TRAIN_CLASSIFY_ENABLE_DATA_AUGUMENTATION) \
            or ((not TRAIN_CLASSIFY_NOT_SEGMENTATION) and TRAIN_SEG_ENABLE_DATA_AUGUMENTATION)
if re_sample:
    r_10 = tumor_records_train[tumor_records_train.diameter_mm < 10.0]
    r_30 = tumor_records_train[(tumor_records_train.diameter_mm >= 10.0) & (tumor_records_train.diameter_mm < 30.0)]
    r_more = tumor_records_train[tumor_records_train.diameter_mm >= 30.0]

    concats = []
    for _ in range(RESAMPLE_DATA_LESS_10_RATIO):
        concats.append(r_10)
    for _ in range(RESAMPLE_DATA_LESS_30_RATIO):
        concats.append(r_30)
    concats.append(r_more)
    tumor_records_train = pd.concat(concats, axis=0)

    print('after resample, got {} samples'.format(tumor_records_train.shape[0]))

# random_offset works iff around_tumor=True
def get_block(record, around_tumor=True, random_offset=(0, 0, 0), shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH)):
    with h5py.File(record['img_numpy_file'], 'r') as hf:
        W, H, D = hf['img'].shape[0], hf['img'].shape[1], hf['img'].shape[2]

        if around_tumor:
            coord = np.array([record['coordX'], record['coordY'], record['coordZ']])
            coord = np.abs((coord - record['origin']) / record['spacing'])
            coord = coord + random_offset
            w, h, d = int(coord[0] - shape[0] // 2), int(coord[1] - shape[1] // 2), int(coord[2] - shape[2] // 2)

            w, h, d = max(w, 0), max(h, 0), max(d, 0)
            w, h, d = min(w, W - shape[0] - 1), min(h, H - shape[1] - 1), min(d, D - shape[2] - 1)
        else:
            w, h, d = random.randint(0, W - shape[0] - 1), random.randint(0, H - shape[1] - 1), random.randint(0, D - shape[2] - 1)

        block = hf['img'][w:w + shape[0], h:h + shape[1], d:d + shape[2]]
        block[block==0] = np.min(hf['img'])
        block = (block - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        block = np.clip(block, 0.0, 1.0)

        # DenseNet paper suggests to (img-mean)/std, use this for simple
        if TRAIN_CLASSIFY_NOT_SEGMENTATION and TRAIN_CLASSIFY_MODEL.lower() == 'densenet':
            block = block - 0.5

        return block

def plot_batch_sample(X, y=None):
    assert X.shape[0] == y.shape[0]

    for b in range(X.shape[0]):
        plot_middle_slices_comparison([X[b, :, :, :, 0], y[b, :, :, :, 0]])

def get_seg_batch(batch_size=32, from_train=True, random_choice=False):
    idx = 0
    records = tumor_records_train if from_train else tumor_records_val

    X = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, INPUT_CHANNEL))
    y = np.zeros((batch_size, INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH, OUTPUT_CHANNEL))

    while True:
        for b in range(batch_size):
            if random_choice:
                idx = random.randint(0, records.shape[0] - 1)

            record = records.iloc[idx]

            is_positive_sample = random.random() < TRAIN_SEG_POSITIVE_SAMPLE_RATIO
            random_offset = np.array([0, 0, 0])
            if TRAIN_SEG_ENABLE_DATA_AUGUMENTATION:
                random_offset = np.array([
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_SEG_SAMPLE_RANDOM_OFFSET, TRAIN_SEG_SAMPLE_RANDOM_OFFSET)
                ])

            X[b,:,:,:,0] = get_block(record, around_tumor=is_positive_sample, random_offset=random_offset,
                                     shape=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH))
            y[b,:,:,:,0] = make_seg_mask(record, create_mask=is_positive_sample, random_offset=random_offset)

            idx = idx + 1 if idx < records.shape[0] - 1 else 0

        # rotate
        if TRAIN_SEG_ENABLE_DATA_AUGUMENTATION:
            for b in range(batch_size):
                _perm = np.random.permutation(3)
                X[b, :, :, :, 0] = np.transpose(X[b, :, :, :, 0], _perm)
                y[b, :, :, :, 0] = np.transpose(y[b, :, :, :, 0], _perm)

        if DEBUG_PLOT_WHEN_GETTING_SEG_BATCH:
            plot_batch_sample(X, y)

        yield X, y

def make_seg_mask(record, create_mask=True, random_offset=(0, 0, 0)):
    mask = np.zeros((INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH))

    if create_mask:
        r = record['diameter_mm'] / 2  + DIAMETER_BUFFER
        radius = np.array([r, r, r])

        if DIAMETER_SPACING_EXPAND:
            radius = radius / record['spacing']

        coord = np.array([INPUT_WIDTH / 2, INPUT_HEIGHT / 2, INPUT_DEPTH / 2])
        coord = coord - random_offset
        radius, coord = radius.astype(np.uint16), coord.astype(np.uint16)

        mask[coord[0] - radius[0]:coord[0] + radius[0] + 1,
             coord[1] - radius[1]:coord[1] + radius[1] + 1,
             coord[2] - radius[2]:coord[2] + radius[2] + 1] = 1.0

    return mask

def get_classify_batch(batch_size=32, from_train=True, random_choice=False):
    idx = 0
    records = tumor_records_train if from_train else tumor_records_val

    shape = (CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH)
    positive_num = int(batch_size * TRAIN_CLASSIFY_POSITIVE_SAMPLE_RATIO)
    X = np.zeros((batch_size, CLASSIFY_INPUT_WIDTH, CLASSIFY_INPUT_HEIGHT, CLASSIFY_INPUT_DEPTH, CLASSIFY_INPUT_CHANNEL))
    y = np.zeros((batch_size, 2))

    while True:
        for b in range(positive_num):
            if random_choice:
                idx = random.randint(0, records.shape[0] - 1)
            record = records.iloc[idx]

            random_offset = np.array([0, 0, 0])
            if TRAIN_CLASSIFY_ENABLE_DATA_AUGUMENTATION:
                random_offset = np.array([
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET),
                    random.randrange(-TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET, TRAIN_CLASSIFY_SAMPLE_RANDOM_OFFSET)
                ])

            X[b, :, :, :, 0] = get_block(record, around_tumor=True, random_offset=random_offset, shape=shape)
            y[b, 0] = 1

            idx = idx + 1 if idx < records.shape[0] - 1 else 0

        for b in range(positive_num, batch_size):
            record = records.iloc[random.randint(0, records.shape[0] - 1)]
            X[b, :, :, :, 0] = get_block(record, around_tumor=False, shape=shape)
            y[b, 1] = 1

        # rotate
        if TRAIN_CLASSIFY_ENABLE_DATA_AUGUMENTATION:
            for b in range(batch_size):
                X[b, :, :, :, 0] = np.transpose(X[b, :, :, :, 0], np.random.permutation(3))

        yield X, y
