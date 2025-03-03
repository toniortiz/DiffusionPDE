# FINISHED
# Replicating the existing Dataset classes for 3d data

"""Streaming 3d data from datasets"""

import os
import numpy as np
import zipfile
import json
import torch
import dnnlib

#----------------------------------------------------------------------------
# Abstract base class for datasets.
# don't need labels, as we are not using them -> Diffusion PDE paper does not use labels either
# See their example call to train the darcy model: torchrun --standalone --nproc_per_node=3 train.py --outdir=pretrained-darcy-new --data=/data/Darcy-merged/ --cond=0 --arch=ddpmpp --batch=60 --batch-gpu=20 --tick=10 --snap=50 --dump=100 --duration=20 --ema=0.05
# -> --cond = 0 means no labels
class DatasetPointCloud(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape                # Shape of the raw data, in the form of num_samples x channels x num_points
    ):
        self._name = name
        self._raw_shape = list(raw_shape)

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_data(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_shape[0]

    def __getitem__(self, idx):
        data = self._load_raw_data(idx)
        assert isinstance(data, np.ndarray)
        assert list(data.shape) == self.data_shape
        assert data.dtype == np.float64
        return data.copy()

    @property
    def name(self):
        return self._name

    @property
    def data_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.data_shape) == 2
        return self.data_shape[0]

    @property
    def num_points(self):
        assert len(self.data_shape) == 2
        return self.data_shape[1]

#----------------------------------------------------------------------------
# Dataset subclass that loads the data recursively from the specified directory
# or ZIP file.

class FolderDatasetPointCloud(DatasetPointCloud):
    def __init__(self,
        path,                   # Path to directory or zip.
        num_points      = None, 
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        else:
            raise IOError('Path must point to a directory')

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npz')
        if len(self._image_fnames) == 0:
            raise IOError('No data files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_data(0).shape)
        if num_points is not None and (raw_shape[1] != num_points):
            raise IOError('Data files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        return None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_data(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            data = np.load(f)
            if fname.endswith('.npz'):
                data = data['arr_0'].astype(np.float64)
            else:
                data = data.astype(np.float64)
        return data

#----------------------------------------------------------------------------
