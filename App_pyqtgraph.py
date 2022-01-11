import sys
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import numpy as np
import pyqtgraph as pg
from pathlib import Path

import random
import time

from pyqtgraph.functions import mkColor, mkPen

pg.setConfigOption('background', (255,255,255,0))
pg.setConfigOption('foreground', (0,0,0))

# # # # # MAIN GUI CLASS DEFINITION # # # # #
class main_GUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        uic.loadUi(r"C:\Users\goliv\Documents\BRAIN\MCE\Acquisition_System\GUI_Qt.ui", self)
        # Enable/disable and link buttons to functions
        self.btn_deactivate.setEnabled(False)
        self.btn_activate.clicked.connect(self.activate)
        self.btn_deactivate.clicked.connect(self.deactivate)
        self.btn_files.clicked.connect(self.browseFiles)
        # Initialize line edits and set default texts
        self.state.setText("Inactive")
        self.video.setFont(QFont('Lato', 12))
        self.video.setText("No video feed")
        self.files_lineEdit.setEnabled(False)
        self.downloads_path = str(Path.home() / "Downloads") # to get the user's downloads Path
        self.camera = 0
        self.empatica_ID = "834ACD"
        self.files_lineEdit.setText("default: " + self.downloads_path)
        self.camera_lineEdit.setPlaceholderText("default: " + str(self.camera))
        self.idE4_lineEdit.setPlaceholderText("default: " + self.empatica_ID)
        # Prepare Threads (not started)
        self.cameraThread = cameraThread()
        self.empaticaThread = empaticaThread()
        self.liveampThread = liveampThread()
        # Build layouts, these are then used to graph
        self.empaticaConstructor = pg.GraphicsLayoutWidget()
        self.eegConstructor = pg.GraphicsLayoutWidget()
        self.empatica_graph.addWidget(self.empaticaConstructor) 
        self.empatica_graph_1 = self.empaticaConstructor.addPlot(row=0, col=0, title = "Temperature")
        self.empatica_graph_2 = self.empaticaConstructor.addPlot(row=1, col=0, title = "Electrodermal Activity")
        self.empatica_graph_3 = self.empaticaConstructor.addPlot(row=2, col=0, title = "Blood Volume Pulse")
        self.empatica_graph_4 = self.empaticaConstructor.addPlot(row=3, col=0, title = "Interbeat Interval")
        self.eeg_graph.addWidget(self.eegConstructor) 
        self.eeg_graph_1 = self.eegConstructor.addPlot(row=0, col=0, title = "EEG Data")
        self.eeg_graph_2 = self.eegConstructor.addPlot(row=1, col=0, title = "EEG Data")
        self.eeg_graph_3 = self.eegConstructor.addPlot(row=2, col=0, title = "EEG Data")
        self.eeg_graph_4 = self.eegConstructor.addPlot(row=3, col=0, title = "EEG Data")
        self.eeg_graph_5 = self.eegConstructor.addPlot(row=0, col=1, title = "EEG Data")
        self.eeg_graph_6 = self.eegConstructor.addPlot(row=1, col=1, title = "EEG Data")
        self.eeg_graph_7 = self.eegConstructor.addPlot(row=2, col=1, title = "EEG Data")
        self.eeg_graph_8 = self.eegConstructor.addPlot(row=3, col=1, title = "EEG Data")

    def browseFiles(self):
        fileName = str(QFileDialog.getExistingDirectory(self, 'Select Directory (folder)'))
        self.files_lineEdit.setText(fileName)

    def ImageUpdateSlot(self, Image):
        self.video.setPixmap(QPixmap.fromImage(Image))

    def activate(self):
        if (self.empatica_checkBox.isChecked() and 
                self.liveamp_checkBox.isChecked() and 
                self.camera_checkBox.isChecked()):
            self.updateControls()
            self.cameraThread.start()
            self.cameraThread.ImageUpdate.connect(self.ImageUpdateSlot)
            self.empaticaThread.start()
            self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)
            self.liveampThread.start()
            self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        elif (self.empatica_checkBox.isChecked() and 
                self.liveamp_checkBox.isChecked() and 
                not(self.camera_checkBox.isChecked())):
            self.updateControls()
            self.empaticaThread.start()
            self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)
            self.liveampThread.start()
            self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        elif (not(self.empatica_checkBox.isChecked()) and 
                self.liveamp_checkBox.isChecked() and 
                self.camera_checkBox.isChecked()):
            self.updateControls()
            self.cameraThread.start()
            self.cameraThread.ImageUpdate.connect(self.ImageUpdateSlot)
            self.liveampThread.start()
            self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        elif (self.empatica_checkBox.isChecked() and 
                not(self.liveamp_checkBox.isChecked()) and 
                self.camera_checkBox.isChecked()):
            self.updateControls()
            self.cameraThread.start()
            self.cameraThread.ImageUpdate.connect(self.ImageUpdateSlot)
            self.empaticaThread.start()
            self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)
        elif (self.empatica_checkBox.isChecked() and 
                not(self.liveamp_checkBox.isChecked()) and 
                not(self.camera_checkBox.isChecked())):
            self.updateControls()
            self.empaticaThread.start()
            self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)
        elif (not(self.empatica_checkBox.isChecked()) and 
                self.liveamp_checkBox.isChecked() and 
                not(self.camera_checkBox.isChecked())):
            self.updateControls()
            self.liveampThread.start()
            self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        elif (not(self.empatica_checkBox.isChecked()) and 
                not(self.liveamp_checkBox.isChecked()) and 
                self.camera_checkBox.isChecked()):
            self.updateControls()
            self.cameraThread.start()
            self.cameraThread.ImageUpdate.connect(self.ImageUpdateSlot)
        else:
            self.noDeviceSelected_popUp()

    # Function to disable functions once the system is activated.     
    def updateControls(self): 
        self.btn_deactivate.setEnabled(True)
        self.btn_activate.setEnabled(False)
        self.btn_files.setEnabled(False)
        self.empatica_checkBox.setEnabled(False)
        self.liveamp_checkBox.setEnabled(False)
        self.camera_checkBox.setEnabled(False)
        self.camera_lineEdit.setEnabled(False)
        self.idE4_lineEdit.setEnabled(False)
        self.state.setText("Active")
    
    # Function to deactivate system.
    def deactivate(self):
        self.btn_deactivate.setEnabled(False)
        self.btn_activate.setEnabled(True)
        self.btn_files.setEnabled(True)
        self.empatica_checkBox.setEnabled(True)
        self.liveamp_checkBox.setEnabled(True)
        self.camera_checkBox.setEnabled(True)
        self.camera_lineEdit.setEnabled(True)
        self.idE4_lineEdit.setEnabled(True)
        self.state.setText("Inactive")
        self.cameraThread.terminate()
        self.empaticaThread.terminate()
        self.liveampThread.terminate()
        saveData()
    
    # This function updates the smartband graph once the system is activated.
    def update_Empatica_Plot(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.empatica_graph_1.clear()
        self.empatica_graph_1.plot(x1, y1, pen=mkPen('y', width = 2))
        self.empatica_graph_2.clear()
        self.empatica_graph_2.plot(x2, y2, pen=mkPen('g', width = 2))
        self.empatica_graph_3.clear()
        self.empatica_graph_3.plot(x3, y3, pen=mkPen('b', width = 2))
        self.empatica_graph_4.clear()
        self.empatica_graph_4.plot(x4, y4, pen=mkPen('r', width = 2))

    # This function updates the EEG cap graphs once the system is activated.
    def update_EEG_plot(self, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8):
        self.eeg_graph_1.clear()
        self.eeg_graph_1.plot(x1, y1, pen=mkPen((169,50,38), width = 2))
        self.eeg_graph_2.clear()
        self.eeg_graph_2.plot(x2, y2, pen= mkPen((202,111,30), width = 2))
        self.eeg_graph_3.clear()
        self.eeg_graph_3.plot(x3, y3, pen=mkPen((212,172,13), width = 2))
        self.eeg_graph_4.clear()
        self.eeg_graph_4.plot(x4, y4, pen=mkPen((40,180,99), width = 2))
        self.eeg_graph_5.clear()
        self.eeg_graph_5.plot(x5, y5, pen=mkPen((25,111,61), width = 2))
        self.eeg_graph_6.clear()
        self.eeg_graph_6.plot(x6, y6, pen=mkPen((93,173,226), width = 2))
        self.eeg_graph_7.clear()
        self.eeg_graph_7.plot(x7, y7, pen=mkPen((46,134,193), width = 2))
        self.eeg_graph_8.clear()
        self.eeg_graph_8.plot(x8, y8, pen=mkPen((125,60,152), width = 2))

    # Pop-up message for one no device is selected.
    def noDeviceSelected_popUp(self):
        self.popUp = QMessageBox()
        self.popUp.setWindowTitle("Warning")
        self.popUp.setText("No device was selected")
        self.popUp.setIcon(QMessageBox.Warning)
        self.popUp.exec_()

# # # # # CLASSES FOR ADDED WIDGETS # # # # #

# Camera thread
class cameraThread(QThread):
    frames_counter = 0
    date_emotion = ()
    emotions_array = []
    cap = 0
    out = 0
    # Read haarcascade document. Let it be know that this document will help identifying the face as quickly
    # as possible. But the emotion detected by this file is NOT the one stored. The emotion stored
    # will be the one detected by the 'dlib' detector backend with the DeepFace.analyze() method.
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ImageUpdate = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cameraThread.cap = cv2.VideoCapture(0)
        # Select codec (.MP4) format and initialize variable that will display the video (out)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        cameraThread.out = cv2.VideoWriter("Hola.mp4", fourcc, 20.0, (640,480))
        while self.ThreadActive:
            ret, frame = cameraThread.cap.read()
            if ret:
                frame2video = frame.copy()
                cv2.putText(frame2video,str(datetime.now()),(380,20), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
                cameraThread.out.write(frame2video)
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                # We set a text in the video where we can visualize the hour with the next line
                cv2.putText(FlippedImage,str(datetime.now()),(380,20), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
                Convert2QtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = Convert2QtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                if ( cameraThread.frames_counter % 200 == 0): #Module of 200 if we have 20 fps, so that every 10 seconds the analysis can be done.
                    # Here the back end analyzes the image and tries to find an emotion
                    try:
                        result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection = True, detector_backend = 'dlib')
                    
                    # In case no emotion is detected, the .analyze() method will throw a ValueError exception.
                    # If we catch it we know there may be no person to be analyzed or the face is not clear
                    # Therefore we attach a 'ND' (Not Detected) value to the emotions array.
                    except ValueError:
                        cameraThread.date_emotion = (datetime.now().isoformat(), 'ND')
                        cameraThread.emotions_array.append(cameraThread.date_emotion)
                        cameraThread.frames_counter += 1
                        continue
                    cameraThread.date_emotion = (datetime.now().isoformat(), result['dominant_emotion'])
                    cameraThread.emotions_array.append(cameraThread.date_emotion)
                cameraThread.frames_counter += 1
                # We will store all emotions in a file every minute. This is a fail-safe measure.
                if (cameraThread.frames_counter % 1200 == 0):
                    with open("hola.csv", 'w', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerow(['Datetime', 'emotion'])
                        writer.writerows(cameraThread.emotions_array)

# Empatica data acquisition
class empaticaThread(QThread):
    # This creates a signal to be sent to the main thread (the GUI)
    update_empatica = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            x1 = np.sort(np.random.randint(low=0, high=15, size=10))
            y1 = np.random.randint(low=0, high=15, size=10)
            x2 = np.sort(np.random.randint(low=0, high=15, size=10))
            y2 = np.random.randint(low=0, high=15, size=10)
            x3 = np.sort(np.random.randint(low=0, high=15, size=10))
            y3 = np.random.randint(low=0, high=15, size=10)
            x4 = np.sort(np.random.randint(low=0, high=15, size=10))
            y4 = np.random.randint(low=0, high=15, size=10)
            # We share data with the main thread using the signal and the .emit() method.
            # Because the main thread is the only one able to graph things. Data can be
            # generated using threads, but any plotting or GUI stuff NEEDS to be done 
            # on the main thread.
            self.update_empatica.emit(x1, y1, x2, y2, x3, y3, x4, y4)
            time.sleep(2)

# EEG data acquisition
class liveampThread(QThread):
    # This creates a signal to be sent to the main thread (the GUI)
    update_EEG = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            x1 = np.sort(np.random.randint(low=0, high=15, size=10))
            y1 = np.random.randint(low=0, high=15, size=10)
            x2 = np.sort(np.random.randint(low=0, high=15, size=10))
            y2 = np.random.randint(low=0, high=15, size=10)
            x3 = np.sort(np.random.randint(low=0, high=15, size=10))
            y3 = np.random.randint(low=0, high=15, size=10)
            x4 = np.sort(np.random.randint(low=0, high=15, size=10))
            y4 = np.random.randint(low=0, high=15, size=10)
            x5 = np.sort(np.random.randint(low=0, high=15, size=10))
            y5 = np.random.randint(low=0, high=15, size=10)
            x6 = np.sort(np.random.randint(low=0, high=15, size=10))
            y6 = np.random.randint(low=0, high=15, size=10)
            x7 = np.sort(np.random.randint(low=0, high=15, size=10))
            y7 = np.random.randint(low=0, high=15, size=10)
            x8 = np.sort(np.random.randint(low=0, high=15, size=10))
            y8 = np.random.randint(low=0, high=15, size=10)
            # We share data with the main thread using the signal and the .emit() method.
            # Because the main thread is the only one able to plot things. Data can be
            # generated using threads, but any plotting or GUI stuff NEEDS to be done 
            # on the main thread.
            self.update_EEG.emit(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8)
            time.sleep(2)

# # # # # GENERAL FUNCTIONS # # # # #
# Method to save data in the end of the execution.
def saveData():
    cameraThread.cap.release()
    cameraThread.out.release()
    with open("hola.csv", 'w', newline = '') as document:
        writer = csv.writer(document)
        writer.writerow(['Datetime', 'emotion'])
        writer.writerows(cameraThread.emotions_array)

# # # # # INITIATE EXECUTION OF APP # # # # #
if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = main_GUI()
    GUI.show()
    sys.exit(app.exec_())


# Matplotlib
# https://stackoverflow.com/questions/36222998/drawing-in-a-matplotlib-widget-in-qtdesigner
# https://www.youtube.com/watch?v=AfEoGSbOSoU
# https://www.youtube.com/watch?v=R_AKgfSMn-E # Easier to understand


# Qt Design Ideas:
# https://www.youtube.com/watch?v=20ed0Ytkxuw