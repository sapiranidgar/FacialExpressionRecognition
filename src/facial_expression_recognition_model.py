from tensorflow.keras import layers, models, optimizers
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class FacialExpressionRecognitionModel:
    def __init__(self):
        self.__model = None

    def build_model(self, number_of_classes: int):
        model = models.Sequential([
            # 1st CNN layer
            layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 2nd CNN layer
            layers.Conv2D(128, (5, 5), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 3rd CNN layer
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # 4th CNN layer
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),

            # Fully connected layers
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),

            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),

            # Output
            layers.Dense(number_of_classes, activation='softmax')
        ])

        # Use modern optimizer argument
        optimizer = optimizers.Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        self.__model = model

    def fit(self, number_of_epochs: int, train_set, test_set):
        checkpoint = ModelCheckpoint(
            filepath="./model.h5",
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_best_only=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        reduce_learningrate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_delta=0.0001,
            verbose=1
        )

        callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

        # 🚀 Let Keras automatically compute steps_per_epoch & validation_steps
        history = self.__model.fit(
            train_set,
            epochs=number_of_epochs,
            validation_data=test_set,
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
