import joblib
import numpy as np
from loguru import logger

from fastapi_skeleton.core.messages import NO_VALID_PAYLOAD
from fastapi_skeleton.models.payload import HousePredictionPayload, payload_to_list
from fastapi_skeleton.models.prediction import HousePredictionResult


class HousePriceModel:
    """
    A class representing a house price prediction model.

    Attributes:
        RESULT_UNIT_FACTOR (int): The factor used to convert the prediction result to a human-readable unit.
        path (str): The path to the saved model file.
        model: The loaded machine learning model.

    Methods:
        __init__(self, path: str) -> None:
            Initializes the HousePriceModel object.

        _load_local_model(self) -> None:
            Loads the machine learning model from the specified path.

        _pre_process(self, payload: HousePredictionPayload) -> np.ndarray:
            Pre-processes the input payload for prediction.

        _post_process(self, prediction: np.ndarray) -> HousePredictionResult:
            Post-processes the prediction result.

        _predict(self, features: np.ndarray) -> np.ndarray:
            Performs the prediction using the loaded model.

        predict(self, payload: HousePredictionPayload) -> HousePredictionResult:
            Predicts the house price based on the input payload.

    """
    RESULT_UNIT_FACTOR = 100

    def __init__(self, path: str) -> None:
        self.path = path
        self._load_local_model()

    def _load_local_model(self) -> None:
        self.model = joblib.load(self.path)

    def _pre_process(self, payload: HousePredictionPayload) -> np.ndarray:
        logger.debug("Pre-processing payload.")
        result = np.asarray(payload_to_list(payload)).reshape(1, -1)
        return result

    def _post_process(self, prediction: np.ndarray) -> HousePredictionResult:
        logger.debug("Post-processing prediction.")
        result = prediction.tolist()
        human_readable_unit = result[0] * self.RESULT_UNIT_FACTOR
        hpp = HousePredictionResult(median_house_value=int(human_readable_unit))
        return hpp

    def _predict(self, features: np.ndarray) -> np.ndarray:
        logger.debug("Predicting.")
        prediction_result = self.model.predict(features)
        return prediction_result

    def predict(self, payload: HousePredictionPayload) -> HousePredictionResult:
        pre_processed_payload = self._pre_process(payload)
        prediction = self._predict(pre_processed_payload)
        logger.info(prediction)
        post_processed_result = self._post_process(prediction)

        return post_processed_result
