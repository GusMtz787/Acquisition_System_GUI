from PyQt5.QtWidgets import *
from datetime import datetime
import os
import _FileStorage as Fs

def browseFiles(self):
    fileName = str(QFileDialog.getExistingDirectory(self, 'Select Directory (folder)'))
    self.files_lineEdit.setText(fileName)
    global downloads_path
    downloads_path = fileName

def activate(self, downloads_path):
    start_datetime = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    video_fileName = os.path.join(downloads_path, (start_datetime + "_VI.mp4"))
    emotions_fileName = os.path.join(downloads_path, (start_datetime + "_FE.csv"))
    eeg_fileName = os.path.join(downloads_path, (start_datetime + "_EEG.csv"))
    Fs.createCSVs(emotions_fileName=emotions_fileName, eeg_fileName=eeg_fileName)
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
