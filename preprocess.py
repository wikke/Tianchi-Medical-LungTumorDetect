import numpy as np
import time
from glob import glob
import SimpleITK as itk
from skimage import morphology, measure, segmentation
import h5py
import _pickle as pickle
from config import *
from visual_utils import plot_slices

if PROCESS_DONE:
    print('done')
    exit()

def preprocess():
    print('start preprocess')
    log_msg("start at {}".format(time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(int(time.time())))))

    ct_files = glob('{}/*/*.mhd'.format(DATASET_PATH))
    handled_ids = set([f[-13:-3] for f in glob('{}/*.h5'.format(PREPROCESS_PATH))])
    print('{} total, {} processed'.format(len(ct_files), len(handled_ids)))

    counter = 0
    for f in ct_files:
        seriesuid = f[-14:-4]
        if seriesuid in handled_ids:
            print('{} handled'.format(seriesuid))
            continue

        counter += 1
        print('{} process {}'.format(counter, f))

        itk_img = itk.ReadImage(f)
        img = itk.GetArrayFromImage(itk_img)  # (depth, height, width)
        img = np.transpose(img, (2, 1, 0))  # (width, height, depth)

        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        _start_time = time.time()
        img, pixels = get_lung_img(img)
        duration = time.time() - _start_time
        cover_ratio = pixels / np.prod(img.shape)

        meta = {
            'seriesuid': seriesuid,
            'shape': img.shape,
            'origin': origin,
            'spacing': spacing,
            'pixels': pixels,
            'cover_ratio': cover_ratio,
            'process_duration': duration,
        }
        save_to_numpy(seriesuid, img, meta)

        log_msg(meta)

    print('all preprocess done')

def log_msg(msg):
    with open(MSG_LOG_FILE, 'a') as f:
        f.write(str(msg) + '\n')
    print(msg)

def save_to_numpy(seriesuid, img, meta):
    file = '{}/{}'.format(PREPROCESS_PATH, seriesuid)

    with h5py.File(file + '.h5', 'w') as hf:
        hf.create_dataset('img', data=img)

    with open(file + '.meta', 'wb') as f:
        pickle.dump(meta, f)

def get_lung_img(img):
    origin_img = img.copy()
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'origin')

    # binary
    img = img < BINARY_THRESHOLD
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'binary')

    # clear_border
    for c in range(img.shape[2]):
        img[:, :, c] = segmentation.clear_border(img[:, :, c])
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'clear_border')

    # keep 2 lagest connected graph
    labels = measure.label(img)
    regions = measure.regionprops(labels)
    labels = [(r.area, r.label) for r in regions]

    if len(labels) > 2:
        labels.sort(reverse=True)
        max_area = labels[2][0]

        for r in regions:
            if r.area < max_area:
                for c in r.coords:
                    img[c[0], c[1], c[2]] = 0
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'keep 2 lagest connected graph')

    # erosion
    # img = morphology.erosion(img, selem=np.ones((2, 2, 2)))
    # if DEBUG_PREPROCESS_PLOT:
    #    plot_slices(img, 'erosion')

    # closing
    img = morphology.closing(img, selem=np.ones((4, 4, 4)))
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'closing')

    # dilation
    img = morphology.dilation(img, selem=np.ones((16, 16, 16)))
    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img, 'dilation')

    if DEBUG_PLOT_WHEN_PREPROCESSING:
        plot_slices(img * origin_img, 'final')

    return img * origin_img, np.sum(img != 0)

if __name__ == '__main__':
    preprocess()
