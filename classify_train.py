from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from config import *
from generators import get_classify_batch
from VGG_model import get_simplified_VGG_classifier, get_full_VGG_classifier
from Inception_model import get_Inception_classifier
from DenseNet_model import get_DenseNet_classifier
import time

def classify_train():
    print('start classify_train')
    if TRAIN_CLASSIFY_MODEL == 'VGG':
        model = get_simplified_VGG_classifier() if USE_SIMPLIFIED_VGG else get_full_VGG_classifier()
    elif TRAIN_CLASSIFY_MODEL == 'Inception':
        model = get_Inception_classifier()
    elif TRAIN_CLASSIFY_MODEL == 'DenseNet':
        model = get_DenseNet_classifier()
    else:
        print('no such model:{}'.format(TRAIN_CLASSIFY_MODEL))
        return

    model.summary()

    run = '{}-{}'.format(time.localtime().tm_hour, time.localtime().tm_min)
    log_dir = CLASSIFY_LOG_DIR.format(run)
    check_point = log_dir + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

    print("classify train round {}".format(run))
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=False)
    checkpoint = ModelCheckpoint(filepath=check_point, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=TRAIN_CLASSIFY_EARLY_STOPPING_PATIENCE, verbose=1)

    model.fit_generator(get_classify_batch(TRAIN_CLASSIFY_TRAIN_BATCH_SIZE, from_train=True), steps_per_epoch=TRAIN_CLASSIFY_STEPS_PER_EPOCH,
                        validation_data=get_classify_batch(TRAIN_CLASSIFY_VALID_BATCH_SIZE, from_train=False), validation_steps=TRAIN_CLASSIFY_VALID_STEPS,
                        epochs=TRAIN_CLASSIFY_EPOCHS, verbose=1,
                        callbacks=[tensorboard, checkpoint, early_stopping])

if __name__ == '__main__':
    classify_train()
