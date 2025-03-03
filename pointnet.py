import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_utils.misc as misc
import dnnlib
import os

# NETWORK STUFF ----------------------------------------------------------------

class PointNetEncoder(torch.nn.Module):
    def __init__(self, channel=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, D, N = x.size()
        pointfeat = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        return torch.cat([x, pointfeat], 1)



class PointNet(torch.nn.Module):
    def __init__(self, num_channels, num_points):
        super().__init__()
        self.feat = PointNetEncoder(num_channels)
        self.lin1 = nn.Linear(1028,512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, num_channels)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.feat(x)
        x = x.permute(0, 2, 1)
        x = self.lin1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.lin2(x))
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn2(x))
        x = x.permute(0,2,1)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# DATA LOADING STUFF -----------------------------------------------------------
    
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
        return self._num_points

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
        if num_points is not None and (raw_shape[0] != num_points):
            raise IOError('Data files do not contain the specified number of points')
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

def main():
    try:
        data_path = "/home/paul_johannssen/Desktop/hiwi job/spatial_model_2/diffusion_pde/DiffusionPDE/data/combined_files/npzs"
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_pointcloud.FolderDatasetPointCloud', path=data_path)
        #dataset_kwargs.num_channels = 5
        dataset_kwargs.num_points = 493
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    except IOError as err:
        exit(f'--data: {err}')

    sampler = torch.utils.data.SequentialSampler(dataset_obj)
    dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=100, sampler=sampler)
    model = PointNet(num_channels=4,num_points=493)  # Assuming you have a defined model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()


    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data in dataloader:
            inputs = data[:,:,:4].float()
            targets = data[:,:,4].float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print("Current loss: ", loss.item(), "Epoch: ", epoch)
            loss.backward()
            optimizer.step()
        
if __name__ == "__main__":
    main()