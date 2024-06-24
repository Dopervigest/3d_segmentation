import functools
import trimesh
import numpy as np
import os
import torch

import matplotlib.pyplot as plt
import pandas as pd

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), vertices

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), vertices

# Use get_raster_points.cache_clear() to clear the cache
@functools.lru_cache(maxsize=4)
def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points

def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
    

def load_intra(name, data_dir='./IntrA3D/obj/', ad_dir = './IntrA3D/ad/'):
    obj_file = data_dir+name
    ad_file = name[:name.index('_')]
    ad_file = ad_file+'-_norm.ad'
    ad_file = ad_dir+ad_file

    labels = np.loadtxt(ad_file)[:, -1].astype(np.int64)  # [segmentation label]
    labels[np.where(labels== 2)] = 1 

    mesh = trimesh.load(obj_file)
    
    return mesh, labels

def load_acdc(name, data_dir):
    obj_file = data_dir+name
    mesh = trimesh.load(obj_file)
    
    colors = mesh.visual.vertex_colors
    tensor = np.array(colors[:,:-1])
    labels = np.argmax(tensor, axis = 1)   
    
    return mesh, labels

def load_clinical(name, data_dir):
    obj_file = data_dir+name
    mesh = trimesh.load(obj_file)
    
    colors = mesh.visual.vertex_colors
    tensor = np.array(colors[:,:-1])
    
    labels = []
    for i in tensor:
        if i[0] == 255:
            labels.append(0)
        elif i[1] == 255:
            labels.append(1)
        elif i[2] == 255:
            labels.append(2)
        else:
            labels.append(3)
    labels = np.array(labels)  
    
    return mesh, labels


def symmetrize(value, mode, dim=None):
    list_ = ["flip", 't', 'cube']

    assert mode in list_#, print(mode)
    
    if mode=='flip':
        assert dim>=2
        weight = 0.5*(value + value.flip(dim))
    elif mode == 't':
        assert type(dim)==list or type(dim)==tuple 
        weight = 0.5*(value + value.transpose(dim[0], dim[1]))
        
    elif mode =='cube':
        assert type(dim)==list or type(dim)==tuple
        weight = 0.5*(value + value.flip(dim[0]))
        weight = 0.5*(weight + weight.flip(dim[1]))
        if len(dim)>2:
            weight = 0.5*(weight + weight.flip(dim[2]))

    return weight


def flip_sym(value, dim:int):
    weight = 0.5*(value + value.flip(dim))
    return weight

def transpose_sym(value, dim:list):
    weight = 0.5*(value + value.transpose(dim[0], dim[1]))
    return weight

def cube_sym_two(value, dim:list):
    weight = value + value.flip(dim[0])
    weight = weight + weight.flip(dim[1])
   
    return weight/3

def cube_sym_three(value, dim:list):
    w1 = (value + value.flip(dim[0]))
    w2 = (w1 + w1.flip(dim[1]))
    w3 = (w2 + w2.flip(dim[2]))
    return w3/4


def equalize_data(list_of_data,list_of_labels, datalen_per_patient = 500, datalen_for_aneurysm = 250):
    arr = np.array([i.numpy() for i in list_of_labels])

    for i in range(len(arr)):
        idx_1 = np.where(arr[i] == np.array([0,1]))[0]
        idx_1 = np.unique(idx_1)

        if len(idx_1) >= datalen_for_aneurysm:
            idx_aneurysm = np.random.choice(idx_1, datalen_for_aneurysm, replace=False)
        else:
            idx_aneurysm = np.random.choice(idx_1, len(idx_1), replace=False)


        idx_0 = np.where(arr[i] == np.array([1,0]))[0]
        idx_0 = np.unique(idx_0)

        num_to_take = datalen_per_patient-len(idx_aneurysm)

        idx_healthy = np.random.choice(idx_0, num_to_take, replace=False)


        final_idx = np.concatenate([idx_aneurysm, idx_healthy])
        np.random.shuffle(final_idx)

        list_of_data[i] = list_of_data[i][final_idx]
        list_of_labels[i] = list_of_labels[i][final_idx]
    
    return list_of_data, list_of_labels

def save_mesh_with_color(mesh, new_colors, path):
    orig_colors = mesh.visual.vertex_colors
    new_colors = torch.argmax(new_colors, dim=1)
    for i in range(len(new_colors)):
        if new_colors[i] == 0:
            orig_colors[i] = [224, 0, 0, 255]
        elif new_colors[i] == 1:
            orig_colors[i] = [0, 224, 0, 255]
        else:
            orig_colors[i] = [0, 0, 224, 255]
        
    mesh.visual.vertex_colors = orig_colors
    file = trimesh.exchange.export.export_mesh(mesh, 'obj', None)
    
    
def save_mesh_with_color_acdc(mesh, new_colors, path):
    orig_colors = mesh.visual.vertex_colors
    for i in range(len(new_colors)):
        if new_colors[i] == 0:
            orig_colors[i] = [224, 0, 0, 255]
        elif new_colors[i] == 1:
            orig_colors[i] = [0, 224, 0, 255]
        else:
            orig_colors[i] = [0, 0, 224, 255]
        
    mesh.visual.vertex_colors = orig_colors
    file = trimesh.exchange.export.export_mesh(mesh, 'obj', None)
    with open(path, mode='w') as f:
        f.write(file)
        
        
def save_mesh_with_color_clinical(mesh, new_colors, path):
    orig_colors = mesh.visual.vertex_colors
    for i in range(len(new_colors)):
        if new_colors[i] == 0:
            orig_colors[i] = [255, 0, 0, 255]
        elif new_colors[i] == 1:
            orig_colors[i] = [0, 255, 0, 255]
        elif new_colors[i] == 2:
            orig_colors[i] = [0, 0, 255, 255]
        else: 
            orig_colors[i] = [150, 150, 150, 255]
        
    mesh.visual.vertex_colors = orig_colors
    file = trimesh.exchange.export.export_mesh(mesh, 'obj', None)
    with open(path, mode='w') as f:
        f.write(file)

def dir_checker(path, name):
    if os.path.exists(f'{path}/{name[:-4]}'):
        print("There's already a dirrectory with this name!")
        for i in range(1000):
            dirname = f'{path}/{name[:-4]} ({i})'
            if not os.path.exists(dirname):
                os.mkdir(dirname)
                break
            else:
                pass
        
    else:
        dirname = f'{path}/{name[:-4]}'
        os.mkdir(dirname)
    
    print(f'Folder for predictions: {dirname}')
    return dirname


def plot_accuracy(path, csv_name, save=True):
    csv_path = path+csv_name
    df = pd.read_csv(csv_path)
    plt.plot(df.epoch, df.accuracy, label='Training accuracy')
    plt.plot(df.epoch, df.val_accuracy, label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if save:
        plt.savefig(path + '/accuracy.png')


def plot_loss(path, csv_name, save=True):
    csv_path = path+csv_name
    df = pd.read_csv(csv_path)
    plt.plot(df.epoch, df.loss, label='Training Loss')
    plt.plot(df.epoch, df.val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save:
        plt.savefig(path + '/loss.png')

    
    
