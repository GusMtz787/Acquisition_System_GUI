from statistics import mode
import sys
import os
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from deepface import DeepFace
from datetime import datetime, timedelta
import csv
import numpy as np
import pyqtgraph as pg
from pathlib import Path
import pandas as pd
import pylsl
import random
import time
import socket

from pyqtgraph.functions import mkColor, mkPen

pg.setConfigOption('background', (255,255,255,0))
pg.setConfigOption('foreground', (0,0,0))

downloads_path = ''
video_fileName = ''
emotions_fileName = ''
eeg_fileName = ''
# inlet variable is declared for the LiveAmp EEG device. It must be global 
# because it is accessed in multiple functions.
inlet = None
empatica_ID = None

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
        global empatica_ID
        empatica_ID = "de6f5a"
        self.files_lineEdit.setPlaceholderText("default: " + downloads_path)
        self.camera_lineEdit.setPlaceholderText("default: " + str(self.camera))
        self.idE4_lineEdit.setPlaceholderText("default: " + empatica_ID)
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
        self.empatica_graph_1.setLabel(axis='left', text='Celsius [°C]')
        self.empatica_graph_2.setLabel(axis='left', text='Microsiemens [µS]')
        self.empatica_graph_3.setLabel(axis='left', text='Nano Watt [nW]')
        self.empatica_graph_4.setLabel(axis='left', text='Seconds [s]')
        self.eeg_graph.addWidget(self.eegConstructor) 
        self.eeg_graph_1 = self.eegConstructor.addPlot(row=0, col=0, title = "EEG Data")
        self.eeg_graph_2 = self.eegConstructor.addPlot(row=1, col=0, title = "EEG Data")
        self.eeg_graph_3 = self.eegConstructor.addPlot(row=2, col=0, title = "EEG Data")
        self.eeg_graph_4 = self.eegConstructor.addPlot(row=3, col=0, title = "EEG Data")
        self.eeg_graph_5 = self.eegConstructor.addPlot(row=0, col=1, title = "EEG Data")
        self.eeg_graph_6 = self.eegConstructor.addPlot(row=1, col=1, title = "EEG Data")
        self.eeg_graph_7 = self.eegConstructor.addPlot(row=2, col=1, title = "EEG Data")
        self.eeg_graph_8 = self.eegConstructor.addPlot(row=3, col=1, title = "EEG Data")
        self.eeg_graph_1.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_2.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_3.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_4.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_5.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_6.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_7.setLabel(axis='left', text='microVolt [µV]')
        self.eeg_graph_8.setLabel(axis='left', text='microVolt [µV]')
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
        start_datetime = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        global video_fileName
        video_fileName = os.path.join(downloads_path, (start_datetime + "_VI.mp4")) 
        global emotions_fileName
        emotions_fileName = os.path.join(downloads_path, (start_datetime + "_FE.csv"))
        global eeg_fileName
        eeg_fileName = os.path.join(downloads_path, (start_datetime + "_EEG.csv"))
        global empatica_fileName
        empatica_fileName = os.path.join(downloads_path, (start_datetime + "_EM.csv"))
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
        # The function receives as parameter a list containing the devices' functions according to the checkboxes from the
        # dictionary. 
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
        if (self.camera_checkBox.isChecked()):
            self.cameraThread.terminate()
            cameraThread.cap.release()
            cameraThread.out.release()
        if (self.empatica_checkBox.isChecked()):
            self.empaticaThread.terminate()
        if (self.liveamp_checkBox.isChecked()):
            self.liveampThread.terminate()
            # The inlet stream from the LiveAmp must be closed, otherwise it won't
            # transmit data if it is executed once more afterwards.
            global inlet
            inletInfo = inlet.info()
            print("Closing {}".format(inletInfo.name))
            inlet.close_stream()
        self.state.setText("Waiting to start")
        self.state.setStyleSheet("color: rgb(0, 133, 199);")
        #saveData()
    
    # This function updates the smartband graph once the system is activated.
    def update_Empatica_Plot(self, y1, y2, y3, y4, y5):
        BVP = pd.DataFrame(y1, columns = ['BVP_timestamp', 'BVP_value'])
        Tem = pd.DataFrame(y2, columns = ['Temperature_timestamp', 'Temperature_value'])
        EDA = pd.DataFrame(y3, columns = ['EDA_timestamp', 'EDA_value'])
        IBI = pd.DataFrame(y4, columns = ['IBI_timestamp', 'IBI_value'])
        Acc = pd.DataFrame(y5, columns = ['Acc_timestamp', 'Acc_value'])

        x1 = np.linspace(0,len(BVP['BVP_value'].to_list())-1,num= len(BVP['BVP_value'].to_list()))     
        x2 = np.linspace(0,len(Tem['Temperature_value'].to_list())-1,num= len(Tem['Temperature_value'].to_list()))            
        x3 = np.linspace(0,len(EDA['EDA_value'].to_list())-1,num= len(EDA['EDA_value'].to_list()))
        x4 = np.linspace(0,len(IBI['IBI_value'].to_list())-1,num= len(IBI['IBI_value'].to_list()))

        self.empatica_graph_1.clear()
        self.empatica_graph_1.plot(x1, BVP['BVP_value'].to_list(), pen=mkPen('y', width = 2))
        self.empatica_graph_2.clear()
        self.empatica_graph_2.plot(x2, Tem['Temperature_value'].to_list(), pen=mkPen('g', width = 2))
        self.empatica_graph_3.clear()
        self.empatica_graph_3.plot(x3, EDA['EDA_value'].to_list(), pen=mkPen('b', width = 2))
        self.empatica_graph_4.clear()
        self.empatica_graph_4.plot(x4, IBI['IBI_value'].to_list(), pen=mkPen('r', width = 2))
    
    # This function will store the values in the EEG csv.
    def update_Empatica_csv(self, y1, y2, y3, y4, y5):
        BVP = pd.DataFrame(y1, columns = ['BVP_timestamp', 'BVP_value'])
        Tem = pd.DataFrame(y2, columns = ['Temperature_timestamp', 'Temperature_value'])
        EDA = pd.DataFrame(y3, columns = ['EDA_timestamp', 'EDA_value'])
        IBI = pd.DataFrame(y4, columns = ['IBI_timestamp', 'IBI_value'])
        Acc = pd.DataFrame(y5, columns = ['Acc_timestamp', 'Acc_value'])

        data = pd.concat([BVP, Tem, EDA, IBI, Acc], axis=1)
        # Use header as false to avoid printing the columns' header each time. 
        data.to_csv(empatica_fileName, mode = 'a', header = False)

    # This function updates the EEG cap graphs once the system is activated.
    def update_EEG_plot(self, channel1_data, channel2_data, channel3_data, channel4_data, 
                        channel5_data, channel6_data, channel7_data, 
                        channel8_data, accx, accy, accz, channel1_numbers, channel2_numbers,
                        channel3_numbers, channel4_numbers, channel5_numbers, channel6_numbers,
                        channel7_numbers, channel8_numbers):
        
        x1 = np.linspace(0,len(channel1_numbers)-1,num= len(channel1_numbers))     
        x2 = np.linspace(0,len(channel2_numbers)-1,num= len(channel2_numbers))            
        x3 = np.linspace(0,len(channel3_numbers)-1,num= len(channel3_numbers))
        x4 = np.linspace(0,len(channel4_numbers)-1,num= len(channel4_numbers))
        x5 = np.linspace(0,len(channel5_numbers)-1,num= len(channel5_numbers))     
        x6 = np.linspace(0,len(channel6_numbers)-1,num= len(channel6_numbers))            
        x7 = np.linspace(0,len(channel7_numbers)-1,num= len(channel7_numbers))
        x8 = np.linspace(0,len(channel8_numbers)-1,num= len(channel8_numbers))

        self.eeg_graph_1.clear()
        self.eeg_graph_1.plot(x1, channel1_numbers, pen=mkPen((169,50,38), width = 2))
        self.eeg_graph_2.clear()
        self.eeg_graph_2.plot(x2, channel2_numbers, pen= mkPen((202,111,30), width = 2))
        self.eeg_graph_3.clear()
        self.eeg_graph_3.plot(x3, channel3_numbers, pen=mkPen((212,172,13), width = 2))
        self.eeg_graph_4.clear()
        self.eeg_graph_4.plot(x4, channel4_numbers, pen=mkPen((40,180,99), width = 2))
        self.eeg_graph_5.clear()
        self.eeg_graph_5.plot(x5, channel5_numbers, pen=mkPen((25,111,61), width = 2))
        self.eeg_graph_6.clear()
        self.eeg_graph_6.plot(x6, channel6_numbers, pen=mkPen((93,173,226), width = 2))
        self.eeg_graph_7.clear()
        self.eeg_graph_7.plot(x7, channel7_numbers, pen=mkPen((46,134,193), width = 2))
        self.eeg_graph_8.clear()
        self.eeg_graph_8.plot(x8, channel8_numbers, pen=mkPen((125,60,152), width = 2))

    # This function will store the values in the EEG csv.
    def update_EEG_csv(self, channel1_data, channel2_data, channel3_data, channel4_data, 
                        channel5_data, channel6_data, channel7_data, 
                        channel8_data, accx, accy, accz):
        data = pd.DataFrame({'channel1_data': pd.Series(channel1_data),
                             'channel2_data': pd.Series(channel2_data),
                             'channel3_data': pd.Series(channel3_data),
                             'channel4_data': pd.Series(channel4_data),
                             'channel5_data': pd.Series(channel5_data),
                             'channel6_data': pd.Series(channel6_data),
                             'channel7_data': pd.Series(channel7_data),
                             'channel8_data': pd.Series(channel8_data),
                             'accx': pd.Series(accx),
                             'accy': pd.Series(accy),
                             'accz': pd.Series(accz)})
        # Use header as false to avoid printing the columns' header each time. 
        data.to_csv(eeg_fileName, mode = 'a', header = False)

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
        # Create emotions CSV file, which will be updated later.
        with open(emotions_fileName, 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Datetime', 'emotion'])

    def eeg_activate(self):
        self.liveampThread.start()
        self.liveampThread.update_EEG.connect(self.update_EEG_plot)
        self.liveampThread.update_EEG.connect(self.update_EEG_csv)    
        # Create EEG CSV file, which will be updated later.
        with open(eeg_fileName, 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['ch1', 'ch2', 'ch3',
                             'ch4', 'ch5', 'ch6',
                             'ch7', 'ch8'])
    
    def empatica_activate(self):
        self.empaticaThread.start()
        self.empaticaThread.update_empatica.connect(self.update_Empatica_Plot)
        self.empaticaThread.update_empatica.connect(self.update_Empatica_csv)
        # Create Empatica CSV file, which will be updated later.
        with open(empatica_fileName, 'w', newline = '') as document:
            writer = csv.writer(document)
            writer.writerow(['Index', 'BVP_timestamp', 'BVP_value', 'Temperature_timestamp', 'Temperature_value', 'EDA_timestamp', 'EDA_value', 'IBI_timestamp', 'IBI_value', 'Acc_timestamp', 'Acc_value'])

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
    update_empatica = pyqtSignal(list, list, list, list, list)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        # SELECT DATA TO STREAM
        acc = True      # 3-axis acceleration
        bvp = True      # Blood Volume Pulse
        gsr = True      # Galvanic Skin Response (Electrodermal Activity)
        tmp = True      # Temperature
        ibi = True

        serverAddress = '127.0.0.1'  #'FW 2.1.0' #'127.0.0.1'
        serverPort = 28000 #28000 #4911
        bufferSize = 4096

        def connect():
            global s
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)

            print("Connecting to server")
            s.connect((serverAddress, serverPort))
            print("Connected to server\n")

            print("Devices available:")
            s.send("device_list\r\n".encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

            print("Connecting to device")
            s.send(("device_connect " + empatica_ID + "\r\n").encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))

            print("Pausing data receiving")
            s.send("pause ON\r\n".encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
            
        connect()

        time.sleep(1)

        def suscribe_to_data():
            if acc:
                print("Suscribing to ACC")
                s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
                response = s.recv(bufferSize)
                print(response.decode("utf-8"))
            if bvp:
                print("Suscribing to BVP")
                s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
                response = s.recv(bufferSize)
                print(response.decode("utf-8"))
            if gsr:
                print("Suscribing to GSR")
                s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
                response = s.recv(bufferSize)
                print(response.decode("utf-8"))
            if tmp:
                print("Suscribing to Temp")
                s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
                response = s.recv(bufferSize)
                print(response.decode("utf-8"))
            if ibi:
                print("Suscribing to Ibi")
                s.send(("device_subscribe " + 'ibi' + " ON\r\n").encode())
                response = s.recv(bufferSize)
                print(response.decode("utf-8"))

            print("Resuming data receiving")
            s.send("pause OFF\r\n".encode())
            response = s.recv(bufferSize)
            print(response.decode("utf-8"))
        
        suscribe_to_data()

        def prepare_LSL_streaming():
            print("Starting LSL streaming")
            if acc:
                infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4')
                global outletACC
                outletACC = pylsl.StreamOutlet(infoACC)
            if bvp:
                infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4')
                global outletBVP
                outletBVP = pylsl.StreamOutlet(infoBVP)
            if gsr:
                infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4')
                global outletGSR
                outletGSR = pylsl.StreamOutlet(infoGSR)
            if tmp:
                infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4')
                global outletTemp
                outletTemp = pylsl.StreamOutlet(infoTemp)
            if ibi:
                infoIbi = pylsl.StreamInfo('ibi','Ibi',1,2,'float32','IBI-empatica_e4')
                global outletIbi
                outletIbi = pylsl.StreamOutlet(infoIbi)
        prepare_LSL_streaming()

        time.sleep(1)

        self.ThreadActive = True
        
        while self.ThreadActive:
            Accelerometers = []
            BVP = []
            Temperature = []
            EDA = []
            IBI = []
            
            print("Streaming...")
            response = s.recv(bufferSize).decode("utf-8")
            samples = response.split("\n") #Variable "samples" contains all the information collected from the wristband.
            print(samples)
            # We need to clean every temporal array before entering the for loop.
            for i in range(len(samples)-1):
                try:
                    stream_type = samples[i].split()[0]
                except:
                    continue
                if (stream_type == "E4_Acc"):
                    timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
                    outletACC.push_sample(data, timestamp=timestamp)
                    timestamp = datetime.fromtimestamp(timestamp)
                    #print(data)#Added in 02/12/20 to show values
                    ACC_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    Accelerometers.append(ACC_tuple)
                if stream_type == "E4_Bvp":
                    timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = float(samples[i].split()[2].replace(',','.'))
                    outletBVP.push_sample([data], timestamp=timestamp)
                    timestamp = datetime.fromtimestamp(timestamp)
                    #print(data)
                    BVP_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    BVP.append(BVP_tuple)
                if stream_type == "E4_Gsr":
                    timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = float(samples[i].split()[2].replace(',','.'))
                    outletGSR.push_sample([data], timestamp=timestamp)
                    timestamp = datetime.fromtimestamp(timestamp)
                    #print(data)
                    GSR_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    EDA.append(GSR_tuple)
                if stream_type == "E4_Temperature":
                    timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = float(samples[i].split()[2].replace(',','.'))
                    outletTemp.push_sample([data], timestamp=timestamp)
                    timestamp = datetime.fromtimestamp(timestamp)
                    #print(data)
                    Temp_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    Temperature.append(Temp_tuple)
                if stream_type == "E4_Ibi":
                    timestamp = float(samples[i].split()[1].replace(',','.'))
                    data = float(samples[i].split()[2].replace(',','.'))
                    outletIbi.push_sample([data], timestamp=timestamp)
                    timestamp = datetime.fromtimestamp(timestamp)
                    #print(data)
                    IBI_tuple = (timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), data)
                    IBI.append(IBI_tuple)

            self.update_empatica.emit(BVP, Temperature, EDA, IBI, Accelerometers)
            time.sleep(1)

# EEG data acquisition
class liveampThread(QThread):
    # This creates a signal to be sent to the main thread (the GUI). And since its original
    # declaration, when update_EEG is called, the values are sent to two functions:
    # update_EEG_plot() and update_EEG_plot(). 
    update_EEG = pyqtSignal(list, list, list, list, list, 
                            list, list, list, list, list, list,
                            list, list, list, list, list, list,
                            list, list)

    # This is the method that is run automatically when the worker is started.
    def run(self):
        self.ThreadActive = True
        # Function to connect EEG to Python LabStream Layer (pylsl)
        def connect_to_EEG():
            stream = pylsl.resolve_stream('type','EEG')
            inlet = pylsl.stream_inlet(stream[0])
            return inlet

        global inlet
        inlet = connect_to_EEG()
        inletInfo = inlet.info()
        print('Connected to:',inletInfo.name(), 'with', inletInfo.channel_count(),'channels. Fs:',inletInfo.nominal_srate())
        outletInfo = pylsl.StreamInfo('eeg', 'EEG', 8, 256, 'int32', '100005-0688')
        outlet = pylsl.StreamOutlet(outletInfo)
        inlet.time_correction()
        date = datetime.now()
        time0 = pylsl.local_clock() - inlet.time_correction()
        
        while self.ThreadActive:
            channel1_data = []
            channel2_data = []
            channel3_data = []
            channel4_data = []
            channel5_data = []
            channel6_data = []
            channel7_data = []
            channel8_data = []
            accx_data = []
            accy_data = []
            accz_data = []
            channel1_numbers = []
            channel2_numbers = []
            channel3_numbers = []
            channel4_numbers = []
            channel5_numbers = []
            channel6_numbers = []
            channel7_numbers = []
            channel8_numbers = []
            sample, time_stamp = inlet.pull_chunk()
            
            if sample != []:
                dates = []
                for i in range(len(time_stamp)-1):
                    timenow = time_stamp[i]-time0
                    updateDate = str(date + timedelta(seconds=timenow))
                    dates.append(updateDate)
                #print(dates)
                for i in range(len(sample)-1):
                    sample_ms = sample[i]
                    channel1_data.append([dates[i],sample_ms[0]])
                    channel2_data.append([dates[i],sample_ms[1]])
                    channel3_data.append([dates[i],sample_ms[2]])
                    channel4_data.append([dates[i],sample_ms[3]])
                    channel5_data.append([dates[i],sample_ms[4]])
                    channel6_data.append([dates[i],sample_ms[5]])
                    channel7_data.append([dates[i],sample_ms[6]])
                    channel8_data.append([dates[i],sample_ms[7]])
                    accx_data.append([dates[i],sample_ms[8]])
                    accy_data.append([dates[i],sample_ms[9]])
                    accz_data.append([dates[i],sample_ms[10]])
                    #vectors to keep only the values of the electrodes, used to graph in real time
                    channel1_numbers.append(int(sample_ms[0]))
                    channel2_numbers.append(int(sample_ms[1]))
                    channel3_numbers.append(int(sample_ms[2]))
                    channel4_numbers.append(int(sample_ms[3]))
                    channel5_numbers.append(int(sample_ms[4]))
                    channel6_numbers.append(int(sample_ms[5]))
                    channel7_numbers.append(int(sample_ms[6]))
                    channel8_numbers.append(int(sample_ms[7]))
            
            # We share data with the main thread using the signal and the .emit() method.
            # Because the main thread is the only one able to plot things. Data can be
            # generated using threads, but any plotting or GUI stuff NEEDS to be done 
            # on the main thread.
            self.update_EEG.emit(channel1_data, channel2_data, channel3_data, channel4_data, 
                                    channel5_data, channel6_data, channel7_data, channel8_data,
                                    accx_data, accy_data, accz_data, channel1_numbers, channel2_numbers,
                                    channel3_numbers, channel4_numbers, channel5_numbers, channel6_numbers,
                                    channel7_numbers, channel8_numbers)
            time.sleep(1)

# # # # # GENERAL FUNCTIONS # # # # #

# Method to save data in the end of the execution.
def saveData():
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

#import cv2
#import numpy as np
#cam=cv2.VideoCapture(1)
#waitTime=50

#while (1):
#    ret,frame=cam.read()
#    cv2.imshow("frame",frame)
#    borderedFrame = cv2.copyMakeBorder(frame,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,200,200])
#    cv2.imshow("bordered frame", borderedFrame)
#    if  cv2.waitKey(waitTime) & 0xFF==ord('q'):
#        break

#cam.release()
#cv2.destroyAllWindows()
# %%
