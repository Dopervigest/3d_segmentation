import torch
from torch.utils.data import Dataset
from scripts import get_surface, get_sdf 
from scripts.utils import scale_to_unit_sphere, load_intra, load_acdc, load_clinical
from scripts.sampling_cube import CubeSampler

import h5py
import torch.utils.data as data

from collections import namedtuple 


class CustomDataset(Dataset):
    def __init__(self, name, device, 
                dataset = 'acdc', # acdc, intra or clinical
                num_classes = None,
                cube_spacing = (16.,16.,16.),
                cube_scale = 1.,
                count = 200,
                train= True,
                scale_range= 0.001,
                shift_sigma= 0,
                rotate_angle = 0.5,
                inference = False,
                return_points = False,
                data_dir='./ACDC_simplified/'):
        
        self.name = name
        self.count = count
        self.device = device
        
        if dataset == 'intra':
            self.num_classes = 2
            mesh, self.colors = load_intra(name)

        elif dataset == 'acdc':
            if not data_dir.endswith('/'):
                data_dir += '/'
            mesh, self.colors = load_acdc(name, data_dir)

        elif dataset == 'clinical':
            if not data_dir.endswith('/'):
                data_dir += '/'
            mesh, self.colors = load_clinical(name, data_dir)

        else:
            raise ValueError("dataset must be 'acdc', 'intra' or 'clinical'")
        self.mesh, self.points = scale_to_unit_sphere(mesh)
        self.surface = get_surface(self.mesh)
        
        if not num_classes:
            print("num_classes wasn't defined, defaulting to 3")
            self.num_classes = 3
        else:
            self.num_classes = num_classes


        if not inference:
            self.cube_sampler = CubeSampler(torch.tensor(self.points), self.colors,
                                    cube_spacing= cube_spacing,
                                    cube_scale= cube_scale,
                                    count=count,
                                    train=train,
                                    scale_range= scale_range,
                                    shift_sigma= shift_sigma,
                                    rotate_angle= rotate_angle,)
            self.permission=False
        else:
            self.cube_sampler = CubeSampler(torch.tensor(self.points), self.colors,
                                    cube_spacing= cube_spacing,
                                    cube_scale= cube_scale,
                                    count=count,
                                    train=train,
                                    scale_range= scale_range,
                                    shift_sigma= shift_sigma,
                                    rotate_angle= rotate_angle,
                                    inference = inference,
                                    return_points = return_points)
            self.permission = True

    def __len__(self):
        return len(self.points) 

    def get_data(self):
        assert self.permission, 'You must create CustomDataset with inference = True to call this function'
        sampling_cube, colors, points = self.cube_sampler.process()       
        
        ret_shape = list(sampling_cube.shape)
        ret_shape[-1]=1
        flat = sampling_cube.view([-1,3]).numpy()
        
        xyz, sdf = get_sdf(self.surface, flat)
        sdf = sdf.reshape(ret_shape)
        sdf = torch.tensor(sdf, device=self.device, dtype=torch.double)
        sdf = sdf.permute([0,4,1,2,3])


        colors = torch.nn.functional.one_hot(torch.tensor(colors, dtype=torch.long), num_classes=self.num_classes).double().to(self.device)

        return sdf, colors,points


    def __getitem__(self, idx):
        
        sampling_cube, colors = self.cube_sampler.process()       
        
        ret_shape = list(sampling_cube.shape)
        ret_shape[-1]=1
        flat = sampling_cube.view([-1,3]).numpy()
        
        xyz, sdf = get_sdf(self.surface, flat)
        sdf = sdf.reshape(ret_shape)
        sdf = torch.tensor(sdf, device=self.device, dtype=torch.double)
        sdf = sdf.permute([0,4,1,2,3])


        colors = torch.nn.functional.one_hot(torch.tensor(colors, dtype=torch.long), num_classes=self.num_classes).double().to(self.device)

        return sdf, colors



class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('points')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):            
        return (torch.from_numpy(self.data[index]).float(),
                torch.from_numpy(self.target[index]).float())

    def __len__(self):
        return self.data.shape[0]
