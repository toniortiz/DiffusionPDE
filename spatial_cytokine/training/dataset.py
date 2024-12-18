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
class Dataset3D(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCDHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        random_seed = 0        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
        # Initialize labels as bunch of zeros
        self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
        self._label_shape = None

    def _get_raw_labels(self):
        # return the zero labels
        return self._raw_labels

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
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        data = self._load_raw_data(raw_idx)
        assert isinstance(data, np.ndarray)
        assert list(data.shape) == self.data_shape
        assert data.dtype == np.float64
        return data.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def data_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.data_shape) == 4 # CDHW
        return self.data_shape[0]

    @property
    def resolution(self):
        assert len(self.data_shape) == 4 # CDHW
        assert self.data_shape[1] == self.data_shape[2] == self.data_shape[3]
        return self.data_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads the data recursively from the specified directory
# or ZIP file.

class FolderDataset3D(Dataset3D):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        # extension is npy right now maybe extend this to npz later if too little storage
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npy')
        if len(self._image_fnames) == 0:
            raise IOError('No data files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_data(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Data files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_data(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            data = np.load(f)
            data = data.astype(np.float64)
        if data.ndim == 3:
            image = image[:, :, :, np.newaxis] # DHW => DHWC
        image = image.transpose(3, 0, 1, 2) # DHWC => CDHW
        return image

#----------------------------------------------------------------------------
