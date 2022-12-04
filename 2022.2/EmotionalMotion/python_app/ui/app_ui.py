import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

from models.LoadAndExtractFeatures import load_data_only_one
from models.PredictEmotion import get_emotion_preedicted_by
from models.SelectMusicByEmotion import select_music_by_emotion
from models.OpenMusicSelectedOnYoutube import open_music_selected_on_youtube
import numpy as np


def create_main_window():
    # create the root window
    root = tk.Tk()
    root.title('Emotional Motion ')
    root.resizable(False, False)
    root.geometry('300x150')
    return root

def select_folder():

    foldername = fd.askdirectory(
        title='Open a directory',
        initialdir='/',
    )
    return foldername


def open_button_setup(root):
    # open button
    open_button = ttk.Button(
        root,
        text='Open a File',
        command=select_folder
    )
    return open_button

def main(root_directory=None):
    one_person = load_data_only_one(root_directory)
    idx = np.random.randint(one_person.shape[0])
    emotion_predicted = int(get_emotion_preedicted_by(one_person[idx, :-1].reshape(1, -1)))
    showinfo(
        title='E-Predicted',
        message=['Negativa', 'Neutra', 'Positiva'][emotion_predicted],
    ).
    music_selected = select_music_by_emotion(str(emotion_predicted))
    open_music_selected_on_youtube(music_selected)


main_window = create_main_window()
open_button = open_button_setup(main_window)
open_button.pack(expand=True)
main(select_folder())


# run the application
main_window.mainloop()
