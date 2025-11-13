from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class FacialExpressionRecognitionModel:
    def __init__(self):
        self.__model = None

    def build_model(self, number_of_classes: int):
        model = Sequential()

        # 1st CNN layer
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 2nd CNN layer
        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 4th CNN layer
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Fully connected 1st layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        # Fully connected layer 2nd layer
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(number_of_classes, activation='softmax'))

        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy'])
        self.__model = model

    def fit(self, number_of_epochs: int, train_set, test_set):
        checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=3,
                                       verbose=1,
                                       restore_best_weights=True
                                       )

        reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=3,
                                                verbose=1,
                                                min_delta=0.0001)

        callbacks_list = [early_stopping, checkpoint, reduce_learningrate]
        history = self.__model.fit(train_set,
                                   steps_per_epoch=train_set.n // train_set.batch_size,
                                   epochs=number_of_epochs,
                                   validation_data=test_set,
                                   validation_steps=test_set.n // test_set.batch_size,
                                   callbacks=callbacks_list
                                   )
        return history

    def plot_model_history(self, history):
        plt.style.use('dark_background')

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.suptitle('Optimizer : Adam', fontsize=10)
        plt.ylabel('Loss', fontsize=16)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.show()