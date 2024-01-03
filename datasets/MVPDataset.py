import os, sys
from .build import DATASETS
import torch
import numpy as np
import torch.utils.data as data
import h5py

@DATASETS.register_module()
class MVP(data.Dataset):
    def __init__(self, config):
        prefix = config.subset
        self.pc_path = config.PC_PATH
        if prefix=="train":
            self.file_path = self.pc_path+'/MVP_Train_CP.h5'
        elif prefix=="test":
            self.file_path = self.pc_path+'/MVP_Test_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        print(f'[DATASET] Open file {self.file_path}')
        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(f'[DATASET] shape {self.input_data.shape}')

        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        self.model_id = np.arange(self.labels.shape[0])
        print(self.gt_data.shape, self.labels.shape, self.model_id.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        model_id = (self.model_id[index])
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, model_id, (partial, complete)
        #return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

        
class MVP_CP16(data.Dataset):
    def __init__(self, train=True, npoints=16384, novel_input=True, novel_input_only=False):
        if train:
            self.input_path = '../data/MVP/mvp_train_input.h5'
            self.gt_path = '../data/MVP/mvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = '../data/MVP/mvp_test_input.h5'
            self.gt_path = '../data/MVP/mvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, partial, complete