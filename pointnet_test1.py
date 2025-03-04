import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_utils.misc as misc
import dnnlib
import os
from training.networks import PointNet
import training.dataset_pointcloud

def main():
    try:
        data_path = "/home/s24pjoha_hpc/DiffusionPDE/data/combined_files/npzs/"#/home/paul_johannssen/Desktop/hiwi job/spatial_model_2/diffusion_pde/DiffusionPDE/data/combined_files/npzs"
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