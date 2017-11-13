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

    ct_files = glob('{}/correct_images/*.mhd'.format(DATASET_PATH))
    meta_data = {}

    for f in ct_files:
        seriesuid = f[-14:-4]

        itk_img = itk.ReadImage(f)
        img = itk.GetArrayFromImage(itk_img) # (depth, height, width)原始数据，不是坐标！！！
        img = np.transpose(img, (1, 2, 0)) # h,w,d

        origin = np.array(itk_img.GetOrigin()) # 元点位置(x,y,z)，世界坐标(mm)
        spacing = np.array(itk_img.GetSpacing()) # 体素之间的间隔(x,y,z)，世界坐标(mm)

        _start_time = time.time()
        img, pixels = get_lung_img(img)
        duration = time.time() - _start_time
        cover_ratio = pixels / np.prod(img.shape)

        save_to_numpy('{}/{}.h5'.format(CT_NUMPY_PATH, seriesuid), img)

        meta_data[seriesuid] = {
            'seriesuid': seriesuid,
            'shape': img.shape,
            'origin': origin,
            'spacing': spacing,
            'pixels': pixels,
            'cover_ratio': cover_ratio,
            'process_duration': duration,
        }
        # log_msg('{},{},{},{},{},{:.4f},{:.2f},saved'.format(seriesuid, img.shape, origin, spacing, pixels, cover_ratio, duration))
        log_msg(meta_data[seriesuid])

    with open(CT_META_FILE, 'wb') as f:
        pickle.dump(meta_data, f)

def log_msg(msg):
    with open(MSG_LOG_FILE, 'a') as f:
        f.write(str(msg) + '\n')

    print(msg)

def save_to_numpy(file, img):
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('img', data=img)

def get_lung_img(img):
    origin_img = img.copy()
    if DEBUG_PREPROCESS_PLOT:
        plot_slices(img, 'origin')

    # binary
    img = img < BINARY_THRESHOLD
    if DEBUG_PREPROCESS_PLOT:
        plot_slices(img, 'binary')

    # clear_border
    for c in range(img.shape[2]):
        img[:, :, c] = segmentation.clear_border(img[:, :, c])
    if DEBUG_PREPROCESS_PLOT:
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
    if DEBUG_PREPROCESS_PLOT:
        plot_slices(img, 'keep 2 lagest connected graph')

    # erosion
    img = morphology.binary_erosion(img, selem=np.ones((2, 2, 2)))
    if DEBUG_PREPROCESS_PLOT:
        plot_slices(img, 'erosion')

    # closing
    img = morphology.binary_closing(img, selem=np.ones((8, 8, 8)))
    if DEBUG_PREPROCESS_PLOT:
        plot_slices(img, 'closing')

    # dilation
    # img = morphology.binary_dilation(img, selem=np.ones((8, 8, 8)))
    # plot_slices(img, 'dilation')

    return img * origin_img, np.sum(img != 0)

if __name__ == '__main__':
    preprocess()
