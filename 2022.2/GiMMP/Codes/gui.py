from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog
import sys, os
import datetime
import matplotlib.pyplot as plt
import qdarktheme
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

        self.model, self.hist = set_model_fit(
            self.model,
            self.train_ds,
            self.epochs,
            self.val_ds
        )

        

        save_model_filename = str(datetime.datetime.now()).replace(' ', '_at_').replace(':', '')
        # os.mkdir(os.path.join(os.getcwd(), 'saved_models'))
        save_model(self.model, 'saved_models/' + save_model_filename + '.hdf5')

        self.data.emit((self.model, self.hist))
        self.finished.emit()


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('gui.ui', self)
        self.initUi()
        self.pb_off()
        self.show()
        self.load_samples_gui()

        self.image_size = (48, 48)
        self.batch_size = 32
        self.num_classes = 7
        self.dsLoaded = False
        self.modelBuit = False
        self.real_label = None
        self.predicted_label = None

    def initUi(self):
        self.pushButtonBuildModel.setEnabled(False)
        self.actionOpen_CK.triggered.connect(self.open_ck)
        self.actionOpen_Target_Face.triggered.connect(self.open_target_face)
        self.actionRestore_model.triggered.connect(self.restore_model)
        self.comboBoxNumberSuggestions.currentIndexChanged[int].connect(
            self.number_suggestions_changed)
        self.pushButtonBuildModel.clicked.connect(self.build_model)
        self.pushButtonTestOnTarget.clicked.connect(self.test_on_target)

    def test_on_target(self):
        self.predicted_label, self.y_pred = predict_on_target(self.model, self.target_path, self.image_size)
        self.label_13.setText("Predicted label: " + self.predicted_label)
        color = "green" if self.real_label == self.predicted_label else "red"
        self.label_13.setStyleSheet("color: " + color + ";")
        self.get_movies()
        self.my_roc_curve()

    def get_movies(self):
        movies = give_me_movies(
            int(self.comboBoxNumberSuggestions.currentText()),
            self.predicted_label)
        self.plainTextEditSuggestions.clear()
        for movie in movies:
            self.plainTextEditSuggestions.insertPlainText(movie[1] + '\n')

    def open_ck(self):
        self.pb_on()
        self.ui_off()
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

    def plot_learning_curves(self):
        train_loss = self.hist.history['loss']
        val_loss = self.hist.history['val_loss']
        train_acc = self.hist.history['accuracy']
        val_acc = self.hist.history['val_accuracy']

        epochs = range(len(train_acc))

        plt.plot(epochs, train_loss,'r', label='train_loss')
        plt.plot(epochs, val_loss,'b', label='val_loss')
        plt.title('train_loss vs val_loss')
        plt.legend()
        plt.figure()

        plt.plot(epochs, train_acc,'r', label='train_acc')
        plt.plot(epochs, val_acc,'b', label='val_acc')
        plt.title('train_acc vs val_acc')
        plt.legend()
        plt.figure()
        plt.show()

    def my_roc_curve(self):
        pass
        # from sklearn.metrics import roc_curve,auc
        # from itertools import cycle
        # import numpy as np

        # new_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        # final_label = new_label
        # new_class = len(new_label)

        # y_pred_ravel = self.y_pred.ravel()
        # lw = 2

        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()

        # y_test = [[-1] * len(new_label)]

        # label_index = int(new_label.index(self.real_label))
        # y_test[0][label_index] = 1


        # print()
        # print(y_test)
        # print()
        # print(self.y_pred)



        # for i in range(new_class):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:,i], self.y_pred[:,i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
            
        # #colors = cycle(['red', 'green','black'])
        # colors = cycle(['red', 'green','black','blue', 'yellow','purple','orange'])
        # for i, color in zip(range(new_class), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
        #              label='ROC curve of class {0}'''.format(final_label[i]))

        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        # plt.xlim([0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")
        # plt.show()


    def open_target_face(self):
        self.target_path = None
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.target_path, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Image File (*.png)", options=options)
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
        self.model_path = None
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.model_path, _ = QFileDialog.getOpenFileName(
            self, "QFileDialog.getOpenFileName()", "", "Model file (*.hdf5)", options=options)
        if self.model_path:
            self.model = restore_model(self.model_path)
            self.modelBuit = True
            self.enable_test_model_on_target_btn()
            self.comboBoxNumberSuggestions.setEnabled(True)

    def set_model(self, obj):
        self.model = obj[0]
        self.hist = obj[1]
        self.plot_learning_curves()

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
            "background-color: rgb(179, 179, 0);")

    def enable_test_model_on_target_btn(self):
        b = (True, "rgb(0, 85, 0)") if self.real_label is not None and self.modelBuit else (False, "rgb(163, 0, 0)")
        self.pushButtonTestOnTarget.setEnabled(b[0])
        self.pushButtonTestOnTarget.setStyleSheet("background-color: " + b[1] + ";")

    def number_suggestions_changed(self, index):
        b = (True, "rgb(0, 85, 0)") if index != 0 and self.dsLoaded else (False, "rgb(179, 179, 0)")
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
        self.im_1 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\angry\\S011_004_00000020.png')
        self.label_1.setPixmap(self.im_1)

        self.im_2 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\disgust\\S011_005_00000019.png')
        self.label_2.setPixmap(self.im_2)

        self.im_3 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\fear\\S050_001_00000016.png')
        self.label_3.setPixmap(self.im_3)

        self.im_4 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\happy\\S010_006_00000014.png')
        self.label_4.setPixmap(self.im_4)

        self.im_5 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\contempt\\S505_002_00000019.png')
        self.label_5.setPixmap(self.im_5)

        self.im_6 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\sad\\S011_002_00000020.png')
        self.label_6.setPixmap(self.im_6)

        self.im_7 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\surprise\\S053_001_00000023.png')
        self.label_7.setPixmap(self.im_7)

        self.im_8 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\angry\\S112_005_00000017.png')
        self.label_8.setPixmap(self.im_8)

        self.im_9 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\fear\\S504_004_00000015.png')
        self.label_9.setPixmap(self.im_9)

        self.im_10 = QPixmap('.\\ckplus\\ck\\CK+48\\train\\happy\\S087_005_00000012.png')
        self.label_10.setPixmap(self.im_10)


app = QtWidgets.QApplication(sys.argv)
app.setStyleSheet(qdarktheme.load_stylesheet())
window = Ui()
app.exec_()


