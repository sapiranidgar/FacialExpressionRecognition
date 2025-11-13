from src.dataset_generator import DatasetGenerator
from src.facial_expression_recognition_model import FacialExpressionRecognitionModel

TRAIN_SET_DIRECTORY_PATH = r"C:\Users\Administrator\Desktop\cyber practice\FacialExpressionRecognition\dataset\train"
TEST_SET_DIRECTORY_PATH = r"C:\Users\Administrator\Desktop\cyber practice\FacialExpressionRecognition\dataset\validation"

if __name__ == "__main__":
    dataset_generator = DatasetGenerator(train_set_directory_path=TRAIN_SET_DIRECTORY_PATH,
                                         test_set_directory_path=TEST_SET_DIRECTORY_PATH,
                                         picture_size=32,
                                         batch_size=128)

    facial_detector = FacialExpressionRecognitionModel()
    facial_detector.build_model(number_of_classes=7)
    history = facial_detector.fit(48, dataset_generator.get_train_set(), dataset_generator.get_test_set())
    facial_detector.plot_model_history(history)
