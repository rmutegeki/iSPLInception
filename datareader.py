# title       :datareader
# description :Script to extract and load the 4 datasets needed for our papers
#              This script is extendable and can accommodate many more datasets
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Either execute the file with "dataset_name" and "dataset_path" specified or call it in utils.py.
# notes       :Uses already downloaded datasets to prepare them for our models
import csv
import glob
import sys

import h5py
import numpy as np
import pandas as pd
import simplejson as json


# Structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data
class DataReader:
    def __init__(self, dataset, datapath, _type='original'):
        if dataset == 'daphnet':
            self.data, self.idToLabel = self._read_daphnet(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity':
            self.data, self.idToLabel = self._read_opportunity(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'pamap2':
            self.data, self.idToLabel = self._read_pamap2(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'ucihar':
            self.data, self.idToLabel = self._read_ucihar(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'ispl':
            self.data, self.idToLabel = self._read_ispl(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        else:
            print('Dataset is not yet supported!')
            sys.exit(0)

    def save_data(self, dataset, path=""):
        f = h5py.File(f'{path}{dataset}.h5', mode='w')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open(f'{path}{dataset}.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.idToLabel))
        print('Done.')

    @property
    def train(self):
        return self.data['train']

    @property
    def validation(self):
        return self.data['validation']

    @property
    def test(self):
        return self.data['test']

    def _read_pamap2(self, datapath):
        files = {
            'train': [
                'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
                'subject107.dat', 'subject108.dat', 'subject109.dat'
            ],
            'validation': [
                'subject105.dat'
            ],
            'test': [
                'subject106.dat'
            ]
        }
        label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            # (24, 'rope jumping')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        # remove the columns we don't need   (Heart rate, temperature, orientation...)
        cols = [
            1,  # Label
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # IMU Hand
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
            38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  # IMU ankle
        ]
        data = {dataset: self._read_pamap2_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_pamap2_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/Protocol/{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the activities with few labeled data and "other"
                    if line[1] == "0" or line[1] == "9" or line[1] == "10" or line[1] == "11" or line[1] == "18" \
                            or line[1] == "19" or line[1] == "20" or line[1] == "24":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) for x in elem[1:]])
                        labels.append(labelToId[elem[0]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def _read_daphnet(self, datapath):
        files = {
            'train': [
                'S01R01.txt', 'S01R02.txt',
                'S03R01.txt', 'S03R02.txt',
                'S06R01.txt', 'S06R02.txt',
                'S07R01.txt', 'S07R02.txt',
                'S08R01.txt', 'S09R01.txt', 'S10R01.txt'
            ],
            'validation': [
                'S02R02.txt', 'S03R03.txt', 'S05R01.txt'
            ],
            'test': [
                'S02R01.txt', 'S04R01.txt', 'S05R02.txt'
            ]
        }
        label_map = [
            # (0, 'Other')
            (1, 'No freeze'),
            (2, 'Freeze')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = {dataset: self._read_daph_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_daph_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the non related activity
                    if line[10] == "0":
                        continue
                    for ind in cols:
                        if ind == 10:
                            if line[ind] == "0":
                                continue
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def _read_opportunity(self, datapath):
        files = {
            'train': [
                'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat',
                'S3-ADL2.dat', 'S3-ADL4.dat', 'S3-ADL5.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-Drill.dat'
            ],
            'validation': [
                'S1-ADL1.dat',
                'S3-ADL3.dat', 'S3-Drill.dat',
                'S4-ADL4.dat',
            ],
            'test': [
                'S2-ADL2.dat', 'S2-Drill.dat',
                'S3-ADL1.dat',
                'S4-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            # (0, 'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            250]
        cols = [x - 1 for x in cols]  # labels for 17 activities (excluding other)

        data = {dataset: self._read_opportunity_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}

        return data, idToLabel

    # this is from https://github.com/nhammerla/deepHAR/tree/master/data and it is an opportunity Challenge reader.
    # It is a python translation for the official one provided by the dataset publishers in Matlab.
    def _read_opportunity_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/dataset/{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the transient activity
                    if line[-1] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    # This data is already windowed and segmented
    def _read_ucihar(self, datapath):
        signals = [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        ]
        label_map = [
            (1, 'Walking'),
            (2, 'Walking_Upstairs'),
            (3, 'Walking_Downstairs'),
            (4, 'Sitting'),
            (5, 'Standing'),
            (6, 'Laying')
        ]
        subjects = {
            # Original train set = 70% of all subjects
            'train': [
                1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17,
                19, 21, 22, 23, 25, 26, 27, 28, 29, 30
            ],
            # 1/3 of test set = 10% of all subjects
            'validation': [
                4, 12, 20
            ],
            # 2/3 of original test set = 20% of all subjects
            'test': [
                2, 9, 10, 13, 18, 24
            ]
        }

        # labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        print('Loading train')
        x_train = self._load_signals(datapath, 'train', signals)
        y_train = self._load_labels(f'{datapath}/train/y_train.txt')
        print('Loading test')
        x_test = self._load_signals(datapath, 'test', signals)
        y_test = self._load_labels(f'{datapath}/test/y_test.txt')
        print("Loading subjects")
        # Pandas dataframes
        subjects_train = self._load_subjects(f'{datapath}/train/subject_train.txt')
        subjects_test = self._load_subjects(f'{datapath}/test/subject_test.txt')

        _data = np.concatenate((x_train, x_test), 0)
        _labels = np.concatenate((y_train, y_test), 0)
        _subjects = np.concatenate((subjects_train, subjects_test), 0)
        print("Data: ", _data.shape, "Targets: ", _labels.shape, "Subjects: ", _subjects.shape)
        data = {dataset: self.split_uci_data(subjects[dataset], _data, _labels, _subjects)
                for dataset in ('train', 'validation', 'test')}

        return data, idToLabel

    def split_uci_data(self, subjectlist, _data, _labels, _subjects):
        data = []
        labels = []
        for i, subject_id in enumerate(subjectlist):
            print(f'Adding Subject {i + 1} -> {subject_id} of {len(subjectlist)} subjects')
            for j, subject in enumerate(_subjects):
                if subject == subject_id:
                    data.append(_data[j])
                    labels.append(_labels[j])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def _load_signals(self, datapath, subset, signals):
        signals_data = []

        for signal in signals:
            filename = f'{datapath}/{subset}/Inertial Signals/{signal}_{subset}.txt'
            signals_data.append(
                pd.read_csv(filename, delim_whitespace=True, header=None).values
            )

        # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
        return np.transpose(signals_data, (1, 2, 0))

    def _load_labels(self, label_path, delimiter=","):
        with open(label_path, 'rb') as file:
            y_ = np.loadtxt(label_path, delimiter=delimiter)
        return y_

    def _load_subjects(self, subject_path, delimiter=","):
        return np.loadtxt(subject_path, delimiter=delimiter)

    def _read_ispl(self, datapath):
        # create the iSPL dataset from raw data in the dataset folder
        datafiles = glob.glob(f"{datapath}/raw/*sensor*.txt")
        files = {
            'train': [
                datafiles[0],
                datafiles[1],
                datafiles[2]
            ],
            'validation': [
                datafiles[3],
                datafiles[4]
            ],
            'test': [
                datafiles[5]
            ]
        }

        label_map = [
            # (0, 'Idle'),
            (1, 'Walking'),
            (2, 'Standing'),
            (3, 'Sitting'),
            # (4, 'Running')
        ]

        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            4, 5, 6,  # Acc x,y,z
            7, 8, 9,  # Gyr x,y,z
            # 10, 11, 12,   # Mag x,y,z
            13, 14, 15,  # lacc x,y,z
            # 16            # Barometer
            0  # ActivityID
        ]

        data = {dataset: self._read_ispl_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_ispl_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for line in reader:
                    elem = []
                    # not including the transient activity
                    if line[0] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _dataset = sys.argv[1]
        _datapath = sys.argv[2]
    else:
        _dataset = input('Enter Dataset name e.g. opportunity, daphnet, ucihar, pamap2:')
        _datapath = input('Enter Dataset root folder: ')
    print(f'Reading {_dataset} from {_datapath}')
    dr = DataReader(_dataset, _datapath)
