from keras.applications import ResNet50
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras import layers, models

from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

import os
# Kill Warning Tensorflow-GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

K.set_image_data_format('channels_last')

from keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
class SensitivitySpecificityCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print('\nPrediction')
            print('Validation testing epoch', epoch)

            # predict model -> true labels = validation_generator.labels
            acc = self.model.evaluate_generator(train_generator(val), steps=np.math.ceil(val.samples / val.batch_size), verbose=False)
            print("Evaluate Accuracy: {:.4f}%".format(acc[3]*100))

            y_pred, x_recon = self.model.predict_generator(train_generator(test), steps=np.math.ceil(test.samples / test.batch_size), verbose=False)
            y_pred = np.argmax(y_pred, axis=1)
            print("Predict Accuracy: {:.4f}%\n".format(accuracy_score(test.labels, y_pred) * 100))

            print(confusion_matrix(list(test.labels), list(y_pred)))
            print(classification_report(list(test.labels), list(y_pred)))

        return self

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def createModel(input_shape, n_class, routings):
    # RESNET
    # best find-tune resnet50 --> 80%
    # adam_fine = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # base_model = Sequential()
    # base_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    # base_model.add(Dense(num_classes, activation='softmax'))
    # # Say not to train first layer (ResNet) model. It is already trained
    # base_model.layers[0].trainable = False
    # return base_model
    # -----------------------------------------------------------------------------------------------

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    # A Resnet Conv2D model
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)
    conv_output = base_model.output # get_layer(name='activation_20')

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv_output, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='softmax'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def DataGenerator(train_batch, val_batch, IMG_SIZE):
    # datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
    #                              samplewise_center=False,  # set each sample mean to 0
    #                              featurewise_std_normalization=False,  # divide inputs by dataset std
    #                              samplewise_std_normalization=False,  # divide each input by its std
    #                              zca_whitening=False,  # apply ZCA whitening
    #                              # zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #                              rotation_range=30,  # randomly rotate images in 0 to 180 degrees
    #                              width_shift_range=0.,  # randomly shift images horizontally
    #                              height_shift_range=0.,  # randomly shift images vertically
    #                              shear_range=0.,  # set range for random shear
    #                              zoom_range=0.,  # set range for random zoom
    #                              channel_shift_range=0.,  # set range for random channel shifts
    #                              # set mode for filling points outside the input boundaries
    #                              fill_mode='nearest',
    #                              cval=0.,  # value used for fill_mode = "constant"
    #                              horizontal_flip=True,  # randomly flip images
    #                              vertical_flip=True,  # randomly flip images
    #                              # set rescaling factor (applied before any other transformation)
    #                              rescale=1./255,
    #                              # image data format, either "channels_first" or "channels_last"
    #                              data_format='channels_last')  # fraction of images reserved for validation (strictly between 0 and 1)

    datagen = ImageDataGenerator(rescale=1. / 255,
                                 data_format='channels_last')  # set as test data

    train_gen = datagen.flow_from_directory('dataset/โรคพืช/train',
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            color_mode='rgb',
                                            class_mode='categorical',
                                            shuffle=True,
                                            batch_size=train_batch) # set as training data

    test_gen = datagen.flow_from_directory('dataset/โรคพืช/test',
                                           target_size=(IMG_SIZE, IMG_SIZE),
                                           color_mode='rgb',
                                           class_mode='categorical',
                                           shuffle=False,
                                           batch_size=val_batch)  # set as validation data

    return train_gen, test_gen, test_gen

def train_generator(generator):
    while 1:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])


if __name__ == '__main__':
    # Hyper parameter
    train_batch = 64
    val_batch = 1
    epochs = 101
    lr = 0.0001
    lam_recon = 0.9  # premitive 0.9 # if lam_recon = 0 remove reconstruction/decoder
    num_classes = 5
    image_size = 28
    resnet_weights_path = './weightPretrain/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # callbacks_list
    filepath = "checkpoints/weights-improvement-{epoch:02d}-{val_capsnet_acc:.3f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_capsnet_acc', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto')
    log = CSVLogger('checkpoints/log.csv')
    callbacks_list = [checkpoint, SensitivitySpecificityCallback(), log]

    train, val, test = DataGenerator(train_batch, val_batch, image_size)

    # Create model
    train_model, eval_model, manipulate_model = createModel((image_size, image_size, 3), n_class=num_classes, routings=3)

    train_model.compile(optimizer=Adam(lr=lr), loss=[margin_loss, 'mse'], loss_weights=[1., lam_recon],
                        metrics={'capsnet': 'accuracy'})
    train_model.summary()

    # train_model.load_weights('checkpoints/weights-improvement-50-0.708.hdf5')

    history = train_model.fit_generator(generator=train_generator(train),
                                        steps_per_epoch=np.math.ceil(train.samples / train_batch),
                                        epochs=epochs,
                                        validation_data=train_generator(val),
                                        validation_steps=np.math.ceil(val.samples / val_batch),
                                        callbacks=callbacks_list,
                                        verbose=2)
