import csv

def createCSVs(emotions_fileName, eeg_fileName):
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