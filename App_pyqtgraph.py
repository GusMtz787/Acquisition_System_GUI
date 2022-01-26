from statistics import mode
import sys
import os
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

downloads_path = ''
start_datetime = ''
video_fileName = ''
emotions_fileName = ''
eeg_fileName = ''

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
        self.state.setText("Waiting to start")
        self.state.setStyleSheet("color: rgb(0, 133, 199);")
        self.video.setFont(QFont('Lato', 12))
        self.video.setText("No video feed")
        self.files_lineEdit.setEnabled(False)
        global downloads_path
        downloads_path = str(Path.home() / "Downloads") # to get the user's downloads Path
        self.camera = 0
        self.empatica_ID = "834ACD"
        self.files_lineEdit.setPlaceholderText("default: " + downloads_path)
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
        self.eeg_graph_1.setDownsampling(mode='peak') # Checar si dejar estos junto con setClipToView
        self.eeg_graph_2.setDownsampling(mode='peak')
        self.eeg_graph_3.setDownsampling(mode='peak')
        self.eeg_graph_4.setDownsampling(mode='peak')
        self.eeg_graph_5.setDownsampling(mode='peak')
        self.eeg_graph_6.setDownsampling(mode='peak')
        self.eeg_graph_7.setDownsampling(mode='peak')
        self.eeg_graph_8.setDownsampling(mode='peak')
        self.eeg_graph_1.setClipToView(True)
        self.eeg_graph_2.setClipToView(True)
        self.eeg_graph_3.setClipToView(True)
        self.eeg_graph_4.setClipToView(True)
        self.eeg_graph_5.setClipToView(True)
        self.eeg_graph_6.setClipToView(True)
        self.eeg_graph_7.setClipToView(True)
        self.eeg_graph_8.setClipToView(True)

    def browseFiles(self):
        fileName = str(QFileDialog.getExistingDirectory(self, 'Select Directory (folder)'))
        self.files_lineEdit.setText(fileName)
        global downloads_path
        downloads_path = fileName

    def ImageUpdateSlot(self, Image):
        self.video.setPixmap(QPixmap.fromImage(Image))

    def activate(self):
        global start_datetime
        global video_fileName
        global emotions_fileName
        global eeg_fileName
        start_datetime = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        video_fileName = os.path.join(downloads_path, (start_datetime + "_VI.mp4")) 
        emotions_fileName = os.path.join(downloads_path, (start_datetime + "_FE.csv"))
        eeg_fileName = os.path.join(downloads_path, (start_datetime + "_EEG.csv"))
        createCSVs()
        self.updateControls()
        # This is a dictionary that will check using the run_functions() which checkboxes are checked, and then it runs
        # the functions according to the selection.
        cases = {
            (True,  True,  True): [self.camera_activate, self.eeg_activate, self.empatica_activate],
            (True,  True,  False): [self.camera_activate, self.eeg_activate],
            (True,  False,  True): [self.camera_activate, self.empatica_activate],
            (True,  False,  False): [self.camera_activate],
            (False,  True, True): [self.eeg_activate, self.empatica_activate],
            (False,  True, False): [self.eeg_activate],
            (False,  False, True): [self.empatica_activate],
            (False,  False, False): [self.noDeviceSelected_popUp]
            }
        self.run_functions(cases[self.camera_checkBox.isChecked(), self.liveamp_checkBox.isChecked(), self.empatica_checkBox.isChecked()])

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
        self.state.setText("Running...")
        self.state.setStyleSheet("color: green;")
    
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
        self.state.setText("Waiting to start")
        self.state.setStyleSheet("color: rgb(0, 133, 199);")
        self.cameraThread.terminate()
        self.empaticaThread.terminate()
        self.liveampThread.terminate()
        saveData()
    
    # This function updates the smartband graph once the system is activated.
    def update_Empatica_Plot(self, y1, y2, y3, y4):
        x1 = np.linspace(0,len(y1)-1,num= len(y1))     
        x2 = np.linspace(0,len(y2)-1,num= len(y2))            
        x3 = np.linspace(0,len(y3)-1,num= len(y3))
        x4 = np.linspace(0,len(y4)-1,num= len(y4))

        self.empatica_graph_1.clear()
        self.empatica_graph_1.plot(x1, y1, pen=mkPen('y', width = 2))
        self.empatica_graph_2.clear()
        self.empatica_graph_2.plot(x2, y2, pen=mkPen('g', width = 2))
        self.empatica_graph_3.clear()
        self.empatica_graph_3.plot(x3, y3, pen=mkPen('b', width = 2))
        self.empatica_graph_4.clear()
        self.empatica_graph_4.plot(x4, y4, pen=mkPen('r', width = 2))

    # This function updates the EEG cap graphs once the system is activated.
    def update_EEG_plot(self, y1, y2, y3, y4, y5, y6, y7, y8):
        x1 = np.linspace(0,len(y1)-1,num= len(y1))     
        x2 = np.linspace(0,len(y2)-1,num= len(y2))            
        x3 = np.linspace(0,len(y3)-1,num= len(y3))
        x4 = np.linspace(0,len(y4)-1,num= len(y4))
        x5 = np.linspace(0,len(y5)-1,num= len(y5))     
        x6 = np.linspace(0,len(y6)-1,num= len(y6))            
        x7 = np.linspace(0,len(y7)-1,num= len(y7))
        x8 = np.linspace(0,len(y8)-1,num= len(y8))

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

    def update_EEG_csv(self, y1, y2, y3, y4, y5, y6, y7, y8):
        # with open(eeg_fileName, 'a', newline = '') as document:
        #     writer = csv.writer(document)
        # El link de abajo es para crear DataFrames con arreglos que tengan
        # diferente tamaño. Falta aplicarlo.
        # https://stackoverflow.com/questions/49891200/generate-a-dataframe-from-list-with-different-length
        pass

    # Pop-up message when no device is selected.
    def noDeviceSelected_popUp(self):
        self.popUp = QMessageBox()
        self.popUp.setWindowTitle("Warning")
        self.popUp.setText("No device was selected")
        self.popUp.setIcon(QMessageBox.Warning)
        self.popUp.exec_()

    # # DEVICES SELECTION FUNCTIONS # #
    def camera_activate(self):
        self.cameraThread.start()
        self.cameraThread.ImageUpdate.connect(self.ImageUpdateSlot)

    def eeg_activate(self):
        self.liveampThread.start()
        self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        self.liveampThread.update_EEG.connect(self.update_EEG_csv)    
    
    def empatica_activate(self):
        self.empaticaThread.start()
        self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)

    def run_functions(self, func_list):
        for function in func_list:
            function()

# # # # # CLASSES FOR DEVICES' FUNCTIONING # # # # #
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
    # Send image to main GUI loop
    ImageUpdate = pyqtSignal(QImage)
    # Initiate camera loop.
    def run(self):
        self.ThreadActive = True
        cameraThread.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Added CAP_DSHOW to avoid warning emerging FOR WINDOWS ONLY!! Remove if any other OS is used.
        # Select codec (.MP4) format and initialize variable that will display the video (out)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cameraThread.out = cv2.VideoWriter(video_fileName, fourcc, 20.0, (640,480))
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
                    with open(emotions_fileName, 'a', newline = '') as document:
                        writer = csv.writer(document)
                        writer.writerows(cameraThread.emotions_array)

# Empatica data acquisition
class empaticaThread(QThread):
    # This creates a signal to be sent to the main thread (the GUI)
    update_empatica = pyqtSignal(list, list, list, list)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            y1 = []
            y2 = []
            y3 = []
            y4 = []
            for _ in range(64):
                y1.append(random.randint(1,20))
            for _ in range(4):  
                y2.append(random.randint(1,20))
                y3.append(random.randint(1,20))
            for _ in range(2):
                y4.append(random.randint(1,20))
            
            # We share data with the main thread using the signal and the .emit() method.
            # Because the main thread is the only one able to graph things. Data can be
            # generated using threads, but any plotting or GUI stuff NEEDS to be done 
            # on the main thread.
            self.update_empatica.emit(y1, y2, y3, y4)
            time.sleep(2)

# EEG data acquisition
class liveampThread(QThread):
    # This creates a signal to be sent to the main thread (the GUI)
    update_EEG = pyqtSignal(list, list, list, list, list, list, list, list)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            y1 = []
            y2 = []
            y3 = []
            y4 = []
            y5 = []
            y6 = []
            y7 = []
            y8 = []
            
            for _ in range(250):
                y1.append(random.randint(1,20))
                y2.append(random.randint(1,20))
                y3.append(random.randint(1,20))
                y4.append(random.randint(1,20))
                y5.append(random.randint(1,20))
                y6.append(random.randint(1,20))
                y7.append(random.randint(1,20))
                y8.append(random.randint(1,20))
            
            # We share data with the main thread using the signal and the .emit() method.
            # Because the main thread is the only one able to plot things. Data can be
            # generated using threads, but any plotting or GUI stuff NEEDS to be done 
            # on the main thread.
            self.update_EEG.emit(y1, y2, y3, y4, y5, y6, y7, y8)
            time.sleep(2)

# # # # # GENERAL FUNCTIONS # # # # #
# Create CSV files where all data will be stored continuosly.
def createCSVs():
        # Create emotions CSV file, which will be updated later.
        with open(emotions_fileName, 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'emotion'])
        # Create EEG CSV file, which will be updated later.
        with open(eeg_fileName, 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Channel_1', 'Channel_2', 'Channel_3'
                                'Channel_4', 'Channel_5', 'Channel_6'
                                    'Channel_7', 'Channel_8'])

# Method to save data in the end of the execution.
def saveData():
    cameraThread.cap.release()
    cameraThread.out.release()
    with open(emotions_fileName, 'a', newline = '') as document:
        writer = csv.writer(document)
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

#%%

def hello():
    print('Jalo')

def go():
    print('Yo también jalo')

# cases = { 
#     (True,  True,  True): [self.camera_activate(), self.eeg_activate(), self.empatica_activate()],
#     (True,  True,  False): [self.camera_activate(), self.eeg_activate()],
#     (True,  False,  True): [self.camera_activate(), self.empatica_activate()],
#     (True,  False,  False): [self.camera_activate()],
#     (False,  True, True): [self.eeg_activate(), self.empatica_activate()],
#     (False,  True, False): [self.eeg_activate()],
#     (False,  False, True): [self.empatica_activate()],
#     (False,  False, False): [self.noDeviceSelected_popUp()],
#     }

cases = { 
    (True,  True,  True): [hello, go],
    (True,  False,  True): 'Hi',
    (True,  True,  False): 'Hi',
    (True,  False,  False): 'Hi',
    (False,  True, True): 'Hi',
    (False,  True, False): 'Hi',
    (False,  False, True): 'Hi',
    (False,  False, False): 'Hi',
    }

x=True
y=True
z=True

def fire_all(func_list):
    for f in func_list:
        f()

fire_all(cases[x,y,z])