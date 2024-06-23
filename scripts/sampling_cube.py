import torch
import numpy as np

device='cpu'

def _rotation_matrix(yaw, pitch, roll):
    
    Rz = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0.0],
            [torch.sin(yaw), torch.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ], device=device
    )
    Ry = torch.tensor(
        [
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0.0, torch.cos(pitch)],
        ], device=device
    )
    Rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(roll), -torch.sin(roll)],
            [0.0, torch.sin(roll), torch.cos(roll)],
        ], device=device
    )

    return Rz @ Ry @ Rx


def _scale_xyz(cube, scale):
    return scale * cube


def _rotate_xyz(cube, M):
    M = _rotation_matrix(M[0], M[1], M[2]).T.double()
    return cube @ M


def _shift_xyz(cube, shift):
    return cube + shift


def _scale_rotate_shift(
    cube : torch.Tensor, scale=torch.tensor([1.2, 1.2, 1.2]),
    rotation=torch.tensor([10.0, 10.0, 10.0]),
    shift=torch.tensor([5.0, 5.0, 5.0])):
    cube = _scale_xyz(cube, scale.float().to(device))
    cube = _rotate_xyz(cube, rotation.float().to(device))
    cube = _shift_xyz(cube, shift.float().to(device))
    return cube


def grid_in_cube(
    spacing=(2, 2, 2), scale=2.0, center_shift=(0.0, 0.0, 0.0)
) -> np.ndarray:
    """Draw samples from the uniform grid that is defined inside a bounding box
    with the center in the `center_shift` and size of `scale`
    Parameters
    ----------
    spacing : tuple, optional
        Number of sections along X, Y, and Z axes (default is (2, 2, 2))
    scale : float, optional
        The scaling factor defines the size of the bounding box (default is 2.)
    center_shift : tuple, optional
        A tuple of ints of coordinates by which to modify the center of the cube (default is (0., 0., 0.))
    Returns
    -------
    ndarray
        3D mesh-grid with shape (spacing[0], spacing[1], spacing[2], 3)
    """

    center_shift_ = np.array(center_shift)
    cube = np.mgrid[
        0 : 1 : spacing[0] * 1j, 0 : 1 : spacing[1] * 1j, 0 : 1 : spacing[2] * 1j
    ].transpose((1, 2, 3, 0))

    return torch.tensor(scale * (cube - 0.5) + center_shift_, device=device)


class CubeSampler():
    def __init__(self, points, colors, cube_spacing = (16.,16.,16.),
        cube_scale = 1.,
        count = 200,
        train= True,
        scale_range= 0.001,
        shift_sigma= 0,
        rotate_angle = 0.5,
        return_points=False,
        inference=False
                ):
        
        self.return_points = return_points
        self.inference = inference
        
        self.points = points
        self.colors = colors
        
        
        self.cube_spacing = cube_spacing
        self.cube_scale = cube_scale
        self.count = count
        self.train = train
        self.scale_range = scale_range
        self.shift_sigma = shift_sigma
        self.rotate_angle = rotate_angle
        
        if inference:
            self.train=False
        
        
    def _make_a_cube(self):
        one_cube = grid_in_cube(
                        spacing=self.cube_spacing, scale=self.cube_scale, center_shift=(0.0, 0.0, 0.0)
                    )

        basic_cube = basic_cube = one_cube.unsqueeze(0).repeat(self.count, 1,1,1,1)

        return basic_cube

    def _get_augmentation_values(self):


        if self.train:
            if self.scale_range == 0:
                scale = torch.ones(self.count, 3)
            else: 
                scale = torch.distributions.uniform.Uniform(
                    low=1.0 - self.scale_range,
                    high=1.0 + self.scale_range).sample((self.count,3))
            if self.rotate_angle ==0:
                rotate = torch.zeros(self.count,3)
            else:
                rotate = torch.distributions.uniform.Uniform(
                    low=0 - self.rotate_angle,
                    high=0 + self.rotate_angle).sample((self.count,3))

            if self.shift_sigma == 0:
                shift = torch.zeros(self.count, 3)
            else:
                shift = torch.normal(1, 0.5, size=(self.count, 3)) * self.shift_sigma

        else:
            scale = torch.ones(self.count, 3)
            rotate = torch.zeros(self.count,3)
            shift = torch.zeros(self.count, 3)


        return scale, rotate, shift
    
    
    def sample_n_points(self, points, colors):
        perm = torch.randperm(points.shape[0])
        idx = perm[:self.count]
        sampled_points = points[idx]
        sampled_colors = colors[idx]
        return sampled_points, sampled_colors
    
    def process(self):
        
        if not self.inference:
            points, colors = self.sample_n_points(self.points, self.colors)
        else:
            points, colors = self.points, self.colors
            self.count = len(points)
        
        basic_cube = self._make_a_cube()
        scale, rotate, shift = self._get_augmentation_values() # get values
        
        shift = torch.add(points, shift.to(device)).squeeze() # shift cube to each point
        

        # no vmap in pytorch
        _xyz_cube = []
        for i in range(len(basic_cube)):
            _xyz_cube.append(_scale_rotate_shift(basic_cube[i], scale[i], rotate[i], shift[i]))

        _xyz_cube = torch.stack(_xyz_cube, dim=0)

        if self.return_points:
            return _xyz_cube, colors, points
        else:
            return _xyz_cube, colors
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
