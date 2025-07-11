# coding: utf-8

"""From the DeepSDF repository https://github.com/facebookresearch/DeepSDF"""

import numpy as np
import plyfile
from skimage.measure import marching_cubes
import time
import torch


def gen_mc_coordinate_grid(N: int, voxel_size: float, t: float = None,
                           device: str = "cpu",
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3

    voxel_size: number
        Size of each voxel

    t: float, optional
        Reconstruction time. Required for space-time models. Default value is
        None, meaning that time is not a model parameter

    device: string, optional
        Device to store tensors. Default is CPU

    voxel_origin: list[number, number, number], optional
        Origin coordinates of the volume. Must be the (bottom, left, down)
        coordinates. Default is [-1, -1, -1]

    Returns
    -------
    samples: torch.Tensor
        A (N**3, 3) shaped tensor with samples' coordinates. If t is not None,
        then the return tensor is has 4 columns instead of 3, with the last
        column equalling `t`.
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    sdf_coord = 3 if t is None else 4
    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device,
                          requires_grad=False)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    # adding the time
    if t is not None:
        samples[:, sdf_coord-1] = t

    return samples


def create_mesh(
    decoder,
    filename="",
    t=None,
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False,
    voxel_origin = [-1, -1, -1],
    voxel_dim = 2.0, 
    resize = [1,1]
    ):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    #voxel_origin = [-1, -1, -1]
    voxel_size = voxel_dim / (N - 1)

    samples = gen_mc_coordinate_grid(
        N, voxel_size, t=t, device=device, voxel_origin=voxel_origin
    )

    sdf_coord = 3 if t is None else 4
   
   
    num_samples = N ** 3
    head = 0

    start = time.time()

    #Define a linear rescalling, can be modified to any function such that alpha(0) = resize[0] and alpha(1) = resize[1]
    def scaling(t): 
        return resize[0] - t*(resize[0]-resize[1]) #+ (torch.sin(t*2*np.pi) + (1-t)*t*torch.cos(t*np.pi/2))/4 for example
        
    
    def rescaled_inr(x):
        t = x[:,-1:]
        xx = x[:,:-1]/(scaling(t))
        zz = torch.cat((xx,t), axis=-1).to(device)
        return scaling(t)*decoder(zz)["model_out"]
    while head < num_samples:
        sample_subset = samples[head:min(head + max_batch, num_samples),
                                0:sdf_coord]
        #decoder(sample_subset)["model_out"]
        samples[head:min(head + max_batch, num_samples), sdf_coord] = (rescaled_inr(sample_subset).squeeze().detach().cpu()
        )
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")
        
        save_ply(np.concatenate((verts,normals),axis=-1)
, faces, filename, vertex_attributes = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")])
        if not silent:
            print("Done")

    return verts, faces, normals, values


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if isinstance(pytorch_3d_sdf_tensor, torch.Tensor):
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.detach().cpu().numpy()
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
    
    if offset is None : 
        level = 0.
    else : 
        level = offset
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print("Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values


def save_ply(
        verts: np.array,
        faces: np.array,
        filename: str,
        vertex_attributes: list = None
) -> None:
    """Converts the vertices and faces into a PLY format, saving the resulting
    file.

    Parameters
    ----------
    verts: np.array
        An NxD matrix with the vertices and its attributes (normals,
        curvatures, etc.). Note that we expect verts to have at least 3
        columns, each corresponding to a vertex coordinate.

    faces: np.array
        An Fx3 matrix with the vertex indices for each triangle.

    filename: str
        Path to the output PLY file.

    vertex_attributes: list of tuples
        A list with the dtypes of vertex attributes other than coordinates.

    Examples
    --------
    > # This creates a simple triangle and saves it to a file called
    > #"triagle.ply"
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > save_ply(verts, faces, "triangle.ply")

    > # Writting normal information as well
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    > attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    > save_ply(verts, faces, "triangle_normals.ply", vertex_attributes=attrs)
    """
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
        #("red","u8") #O-255
    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if vertex_attributes is not None:
        dtypes[3:3] = vertex_attributes

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)

