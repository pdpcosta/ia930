from joblib import load
from classifier_model_RFC.MODEL_CONST import MODEL_DIRECTORY
import sklearn


def get_emotion_preedicted_by(user_data_sensor):
    pred = load(MODEL_DIRECTORY())
    return pred.predict(user_data_sensor)
