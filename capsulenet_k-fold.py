import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from capsulenet import createModel, margin_loss
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from keras.backend import clear_session
import os
# Kill Warning Tensorflow-GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generator2array(generator):
    batches = zip(*(next(generator) for _ in range(len(generator))))
    # concatenate all inputs and outputs, respectively.
    x, y = (np.concatenate(b) for b in batches)
    return x, y


PATH_DIR = 'D:/Project/Ricefit/dataset/โรคพืช/raw_data'
IMG_SIZE = 28
num_classes = 6
lr = 0.00001
train_batch = 64
epochs = 50
data_raw = ImageDataGenerator().flow_from_directory(PATH_DIR, class_mode='categorical', target_size=(IMG_SIZE, IMG_SIZE), shuffle=True, seed=42)
X, y = generator2array(data_raw)

################################ Fix Problem ###############################################
img_name = []
batches_per_epoch = data_raw.samples // data_raw.batch_size + (data_raw.samples % data_raw.batch_size > 0)
current_index = 0
for i in range(batches_per_epoch):
    index_array = data_raw.index_array[current_index:current_index+data_raw.batch_size]
    img_name = img_name + [data_raw.filenames[idx] for idx in index_array]
    current_index = current_index + 32

data_raw.reset()
#####################################################################################################

print(data_raw.class_indices)
kf = KFold(n_splits=10, shuffle=False)
cvscores, count = [], 1
for train, test in kf.split(X, y):
    (X_train, y_train), (X_test, y_test) = (X[train], y[train]), (X[test], y[test])
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Information train-test
    print('Train Number', Counter(np.argmax(y[train], axis=1)))
    print('Test Number', Counter(np.argmax(y[test], axis=1)))

    # callbacks_list
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{val_capsnet_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_capsnet_acc', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto')
    log = CSVLogger('checkpoints/log.csv')
    callbacks_list = [checkpoint, log]

    train_model, eval_model, manipulate_model = createModel((IMG_SIZE, IMG_SIZE, 3), n_class=num_classes, routings=3)
    train_model.compile(optimizer=Adam(lr=lr), loss=[margin_loss, 'mse'], loss_weights=[1., 0.392], metrics={'capsnet': 'accuracy'})
    # train_model.summary()

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    train_model.fit_generator(generator=train_generator(X_train, y_train, train_batch, 0.1),
                              steps_per_epoch=int(len(X_train) / train_batch),
                              epochs=epochs,
                              validation_data=([X_test, y_test], [y_test, X_test]),
                              verbose=2)
    train_model.save_weights('weight'+str(count)+'.h5')
    y_pred, x_recon = train_model.predict([X_test, y_test], verbose=2)
    scores = (np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0]) * 100
    cvscores.append(scores)

    print("Predict Accuracy: {:.4f}%\n".format(scores))
    print(confusion_matrix(list(np.argmax(y_test, 1)), list(np.argmax(y_pred, 1))))
    print(classification_report(list(np.argmax(y_test, 1)), list(np.argmax(y_pred, 1))))
    clear_session()

    #####################################################################################################
    import pandas as pd
    results = pd.DataFrame({"Filename": np.array(img_name)[test],
                            "Actuals": list(np.argmax(y_test, 1)),
                            "Predictions": list(np.argmax(y_pred, 1))})
    results.to_excel('result' + str(count) + '.xlsx')
    count = count + 1
    #####################################################################################################

print("{:.2f}% % (+/- {:.2f}%)".format(np.mean(cvscores), np.std(cvscores)))
