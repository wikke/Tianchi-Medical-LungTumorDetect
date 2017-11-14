from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from config import *
from generators import get_classify_batch
from classify_model import get_VGG_classifier
import time

def classify_train():
    print('start classify_train')
    model = get_VGG_classifier()
    model.summary()

    run = '{}-{}'.format(time.localtime().tm_hour, time.localtime().tm_min)

    print("classify train round {}".format(run))
    tensorboard = TensorBoard(log_dir=CLASSIFY_LOG_DIR.format(run), histogram_freq=0, write_grads=False, write_graph=False)
    checkpoint = ModelCheckpoint(filepath=CLASSIFY_LOG_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=TRAIN_CLASSIFY_EARLY_STOPPING_PATIENCE, verbose=1)

    model.fit_generator(get_classify_batch(TRAIN_CLASSIFY_TRAIN_BATCH_SIZE), steps_per_epoch=TRAIN_CLASSIFY_STEPS_PER_EPOCH,
                        validation_data=get_classify_batch(TRAIN_CLASSIFY_VALID_BATCH_SIZE), validation_steps=TRAIN_CLASSIFY_VALID_STEPS,
                        epochs=TRAIN_CLASSIFY_EPOCHS, verbose=1,
                        callbacks=[tensorboard, checkpoint, early_stopping])

if __name__ == '__main__':
    classify_train()
