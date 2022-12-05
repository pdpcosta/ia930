import random

from models.ReadCSVFile import read_csv_file
from music_data.MUSIC_DATA_CONST import GOOD_EMOTIONS_MUSIC_DIRECTORY, NEUTRAL_EMOTIONS_MUSIC_DIRECTORY


def select_music_by_emotion(emotion_predicted):
    if emotion_predicted == ("0" or "1"):
        good_emotions_musics = read_csv_file(GOOD_EMOTIONS_MUSIC_DIRECTORY())
        music_selected = random.choice(good_emotions_musics)
        return music_selected
    else:
        neutral_emotions_musics = read_csv_file(NEUTRAL_EMOTIONS_MUSIC_DIRECTORY())
        music_selected = random.choice(neutral_emotions_musics)
        return music_selected
