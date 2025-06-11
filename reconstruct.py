#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""

import argparse
import os
import os.path as osp
import torch
from src.model import from_pth
from src.util import reconstruct_at_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run marching cubes using a trained model."
    )
    parser.add_argument(
        "model_path",
        help="Path to the PyTorch weights file"
    )
    parser.add_argument(
        "output_path",
        help="Path to the output mesh file"
    )
    parser.add_argument(
        "--omega0", "-w", type=int, default=1,
        help="Value for \\omega_0. Default is 1"
    )
    parser.add_argument(
        "--resolution", "-r", default=128, type=int,
        help="Resolution to use on marching cubes. Default is 128"
    )
    parser.add_argument(
        "--times", "-t", nargs='+', default=[-1, 0, 1],
        help="Parameter values to run inference on. Default is [-1, 0, 1]."
        "Use None value to render sdf with no time"
        "Use linspace k to render from time as linspace(0,1,k)" 
    )

    parser.add_argument(
        "--lipschitz", "-l", default = False, action = "store_true", 
        help = "Perform reconstruction of lipmlp")
    parser.add_argument(
        "--name", default = "mesh", help ="give a specific name when rendering sdf")
    
    parser.add_argument(
        "--scale", "-s", nargs='+', default=[1, 1], type = float, 
        help="Parameter scale to rescale the shape, one value for each shape. IF the value is s<0, the scaling will be -1/s. "
    ) 
    
    args = parser.parse_args()
    out_dir = osp.split(args.output_path)[0]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    devstr = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(devstr)

    dd = torch.load(args.model_path)

    model = from_pth(args.model_path, w0=args.omega0, device=device, lip=args.lipschitz).eval()
    model = model.to(device)
    print(model)

    print(f"Running marching cubes running with resolution {args.resolution}")
    if args.times[0] =="None" :
        times = [None]
    elif args.times[0] =="linspace":
        times = [t.item() for t in torch.linspace(0.,1., int(args.times[1]))]
    else  : times = [float(t) for t in args.times]
    
    if args.scale[0]<0 : args.scale[0] = -1/args.scale[0]
    if args.scale[1]<0 : args.scale[1] = -1/args.scale[1]
    reconstruct_at_times(model, times, out_dir, resolution=args.resolution, device=device, name=args.name,resize = args.scale )#, voxel_origin = [-2, -2, -2], voxel_dim=4)

    print("Done")
