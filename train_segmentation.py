from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from model_UNet import get_unet, UNetEvaluator
from config import *
from generators import get_seg_batch
import time

def seg_train():
    print('start seg_train')
    model = get_unet()
    model.summary()

    run = '{}-{}'.format(time.localtime().tm_hour, time.localtime().tm_min)
    log_dir = SEG_LOG_DIR.format(run)
    check_point = log_dir + '/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5'

    print("seg train round {}".format(run))
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=False)
    checkpoint = ModelCheckpoint(filepath=check_point, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=TRAIN_SEG_EARLY_STOPPING_PATIENCE, verbose=1)
    evaluator = UNetEvaluator()
    model.fit_generator(get_seg_batch(TRAIN_SEG_TRAIN_BATCH_SIZE, from_train=True), steps_per_epoch=TRAIN_SEG_STEPS_PER_EPOCH,
                        validation_data=get_seg_batch(TRAIN_SEG_VALID_BATCH_SIZE, from_train=False), validation_steps=TRAIN_SEG_VALID_STEPS,
                        epochs=TRAIN_SEG_EPOCHS, verbose=2,
                        callbacks=[tensorboard, checkpoint, early_stopping, evaluator])

if __name__ == '__main__':
    seg_train()
