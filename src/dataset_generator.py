from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetGenerator:
    def __init__(self, train_set_directory_path: str, test_set_directory_path: str, picture_size: int, batch_size: int):
        datagen_train = ImageDataGenerator()
        datagen_val = ImageDataGenerator()

        self.__train_set = datagen_train.flow_from_directory(train_set_directory_path,
                                                             target_size=(picture_size, picture_size),
                                                             color_mode="grayscale",
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             shuffle=True)

        self.__test_set = datagen_val.flow_from_directory(test_set_directory_path,
                                                          target_size=(picture_size, picture_size),
                                                          color_mode="grayscale",
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          shuffle=False)

    def get_train_set(self):
        return self.__train_set

    def get_test_set(self):
        return self.__test_set
