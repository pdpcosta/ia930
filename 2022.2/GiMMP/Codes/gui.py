from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import sys, os
import datetime
from main import (load_images, make_model, set_model_compile, set_model_fit, 
    predict_on_target, restore_model, save_model, give_me_movies)


class WorkerLoadImages(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(object)

    def run(self):
        train, val = load_images()
        self.finished.emit()
        self.data.emit((train, val))


class WorkerTrainModel(QObject):
    finished = pyqtSignal()
    data = pyqtSignal(object)

    def __init__(self, train_ds, val_ds, image_size, num_classes, augment_data, epochs):
        super(WorkerTrainModel, self).__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.image_size = image_size
        self.num_classes = num_classes
        self.augment_data = augment_data
        self.epochs = epochs

    def run(self):
        self.train_ds = self.train_ds.prefetch(buffer_size=32)
        self.val_ds = self.val_ds.prefetch(buffer_size=32)
        
        self.model = make_model(
            input_shape=self.image_size + (3,),
            num_classes=self.num_classes,
            augment_data=self.augment_data
            )

        self.model = set_model_compile(self.model)

        self.model = set_model_fit(
            self.model,
            self.train_ds,
            self.epochs,
            self.val_ds
        )

        

        save_model_filename = str(datetime.datetime.now()).replace(' ', '_at_').replace(':', '')
        # os.mkdir(os.path.join(os.getcwd(), 'saved_models'))
        save_model(self.model, 'saved_models/' + save_model_filename + '.hdf5')

        self.finished.emit()
        self.data.emit(self.model)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gui.ui', self)
        self.initUi()
        self.pb_off()
        self.show()

        self.image_size = (48, 48)
        self.batch_size = 32
        self.num_classes = 7
        self.dsLoaded = False
        self.modelBuit = False
        self.real_label = None
        self.predicted_label = None

    def initUi(self):
        self.pushButtonBuildModel.setEnabled(False)
        self.actionOpen_FER.triggered.connect(self.open_fer)
        self.actionOpen_Target_Face.triggered.connect(self.open_target_face)
        self.actionRestore_model.triggered.connect(self.restore_model)
        self.comboBoxNumberSuggestions.currentIndexChanged[int].connect(
            self.number_suggestions_changed)
        self.pushButtonBuildModel.clicked.connect(self.build_model)
        self.pushButtonTestOnTarget.clicked.connect(self.test_on_target)

    def test_on_target(self):
        self.predicted_label = predict_on_target(self.model, self.target_path, self.image_size)
        self.label_13.setText("Predicted label: " + self.predicted_label)
        color = "green" if self.real_label == self.predicted_label else "red"
        self.label_13.setStyleSheet("color: " + color + ";")
        self.get_movies()

    def get_movies(self):
        movies = give_me_movies(
            int(self.comboBoxNumberSuggestions.currentText()),
            self.predicted_label)
        self.plainTextEditSuggestions.clear()
        for movie in movies:
            self.plainTextEditSuggestions.insertPlainText(movie[1] + '\n')

    def open_fer(self):
        self.pb_on()
        self.ui_off()
        self.load_samples_gui()
        self.thread = QThread()
        self.worker = WorkerLoadImages()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.data.connect(self.set_data)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.load_images_finished)
        self.thread.start()

    def open_target_face(self):
        self.target_path = None
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.target_path, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Image File (*.jpg)", options=options)
        if self.target_path:
            self.im_target = QPixmap(self.target_path)
            self.label_11.setPixmap(self.im_target)
            self.label_12.setText("Real label: " + self.target_path.split("/")[-2].capitalize())
            self.label_13.setText("Predicted label: ???")
            self.real_label = self.target_path.split("/")[-2].capitalize()
            self.enable_test_model_on_target_btn()

    def build_model(self):
        self.pb_on()
        self.ui_off()
        self.thread = QThread()
        self.worker = WorkerTrainModel(
            self.train_ds,
            self.val_ds,
            self.image_size,
            self.num_classes,
            self.checkBoxDataAugmentation.isChecked(),
            self.spinBoxEpoch.value()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.data.connect(self.set_model)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.build_model_finished)
        self.thread.start()

    def restore_model(self):
        # self.model_path = None
        # self.model_path = QFileDialog.getExistingDirectory(
        #     self, "Select the model folder")
        # if self.model_path:
        #     self.model = restore_model(self.model_path)
        #     self.modelBuit = True
        #     self.enable_test_model_on_target_btn()

        self.model_path = None
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.model_path, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Model file (*.hdf5)", options=options)
        if self.model_path:
            self.model = restore_model(self.model_path)
            self.modelBuit = True
            self.enable_test_model_on_target_btn()

    def set_model(self, obj):
        self.model = obj

    def build_model_finished(self):
        self.pb_off()
        self.ui_on()
        self.modelBuit = True
        self.enable_test_model_on_target_btn()

    def set_data(self, obj):
        self.train_ds = obj[0]
        self.val_ds = obj[1]

    def load_images_finished(self):
        self.pb_off()
        self.ui_on()
        self.dsLoaded = True
        self.comboBoxNumberSuggestions.setEnabled(True)
        self.pushButtonBuildModel.setStyleSheet(
            "background-color: yellow;")

    def enable_test_model_on_target_btn(self):
        b = (True, "green") if self.real_label is not None and self.modelBuit else (False, "red")
        self.pushButtonTestOnTarget.setEnabled(b[0])
        self.pushButtonTestOnTarget.setStyleSheet("background-color: " + b[1] + ";")

    def number_suggestions_changed(self, index):
        b = (True, "green") if index != 0 and self.dsLoaded else (False, "yellow")
        self.pushButtonBuildModel.setEnabled(b[0])
        self.pushButtonBuildModel.setStyleSheet("background-color: " + b[1] + ";")

    def pb_on(self):
        self.progressBar.setVisible(True)

    def pb_off(self):
        self.progressBar.setVisible(False)

    def ui_off(self):
        self.setEnabled(False)

    def ui_on(self):
        self.setEnabled(True)

    def load_samples_gui(self):
        self.im_1 = QPixmap('.\\FER2013\\train\\angry\\Training_3908.jpg')
        self.label_1.setPixmap(self.im_1)

        self.im_2 = QPixmap('.\\FER2013\\train\\disgust\\Training_5420780.jpg')
        self.label_2.setPixmap(self.im_2)

        self.im_3 = QPixmap('.\\FER2013\\train\\fear\\Training_358270.jpg')
        self.label_3.setPixmap(self.im_3)

        self.im_4 = QPixmap('.\\FER2013\\train\\happy\\Training_177442.jpg')
        self.label_4.setPixmap(self.im_4)

        self.im_5 = QPixmap('.\\FER2013\\train\\neutral\\Training_780242.jpg')
        self.label_5.setPixmap(self.im_5)

        self.im_6 = QPixmap('.\\FER2013\\train\\sad\\Training_5682645.jpg')
        self.label_6.setPixmap(self.im_6)

        self.im_7 = QPixmap('.\\FER2013\\train\\surprise\\Training_5070080.jpg')
        self.label_7.setPixmap(self.im_7)

        self.im_8 = QPixmap('.\\FER2013\\train\\angry\\Training_3908.jpg')
        self.label_8.setPixmap(self.im_8)

        self.im_9 = QPixmap('.\\FER2013\\train\\fear\\Training_698141.jpg')
        self.label_9.setPixmap(self.im_9)

        self.im_10 = QPixmap('.\\FER2013\\train\\happy\\Training_3815265.jpg')
        self.label_10.setPixmap(self.im_10)


app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()

