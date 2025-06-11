#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import math
import os
import os.path as osp
import time
import sys
try:
    import kaolin
except ImportError:
    KAOLIN_AVAILABLE = False
else:
    KAOLIN_AVAILABLE = True
import numpy as np
import torch
import yaml
from src.dataset import _read_ply
from src.model import SIREN
from scipy.spatial import KDTree


def distance(sample,pc,device):
    kdtree = KDTree(pc.cpu())
    dist, index = kdtree.query(sample.cpu().detach(), 1)
    return torch.tensor(dist,device=device), torch.tensor(index,device=device)

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def set_opt_lr_value(opt,lr):
    for g in opt.param_groups:
        g['lr'] = lr

def sample_pc(pc, n1,n2,n3 ,device , ext1 = 6, ext2=2):
    m1 = ext1*torch.ones(3,device=device)
    p1 = ext1/2*torch.ones(3,device=device)
    m2 = ext2*torch.ones(3,device=device)
    p2 = ext2/2*torch.ones(3,device=device)

    choice = np.random.choice(np.arange(pc.shape[0]), n1)
    pc_surface = pc[choice,:].requires_grad_(True)

    
    pc_ext_f_t = (torch.rand(n2,3,device=device)*m1-p1).requires_grad_(True)
    mask = abs(pc_ext_f_t).max(axis=-1)[0]>1.
    pc_ext_f = pc_ext_f_t[mask].clone()
    while pc_ext_f.shape[0]<n2 : 
        pc_ext_f_t = (torch.rand(n2,3,device=device)*m1-p1).requires_grad_(True)
        mask = abs(pc_ext_f_t).max(axis=-1)[0]>1.
        pc_ext_f = torch.cat((pc_ext_f, pc_ext_f_t[mask]))
    pc_ext_c = (torch.rand(n3,3,device=device)*m2-p2).requires_grad_(True)
    return pc_surface,choice,pc_ext_c, pc_ext_f

def loss_sdf(sdf,pc_ext_c, pc_ext_f,pc_surface,choice,pc,normal,device):
    out_ext = sdf(pc_ext_c)
    pred_ext = out_ext["model_out"]
    ### Changer ca pour un kd-tree. 
    dist,ind_dist  = distance(pc_ext_c,pc,device)
    with torch.no_grad():
      ind_in = torch.sum((pc_ext_c-pc[ind_dist])*(normal[ind_dist]),axis=1).sign()
      sign_d = ind_in*dist
    loss_ext = torch.linalg.norm(pred_ext-sign_d.view(pred_ext.shape[0],1))**2/(pred_ext.shape[0])

    out_surf = sdf(pc_surface.float())
    pred_surface = out_surf["model_out"]
    loss_surface_ci = torch.linalg.norm(pred_surface)**2/(pred_surface.shape[0])


    gradf = gradient(pred_surface,out_surf["model_in"])
    grad_ext = gradient(pred_ext, out_ext["model_in"])

    loss_ext_eik = torch.linalg.norm(1-torch.linalg.norm(grad_ext,axis=1))**2/gradf.shape[0]
    
    loss_surface_eik = torch.linalg.norm(1-torch.linalg.norm(gradf,axis=1))**2/gradf.shape[0]


    scalar = (torch.sum(gradf * normal[choice], dim=-1))
    loss_surface_ne = torch.linalg.norm(1-scalar)**2/gradf.shape[0]

    pred_far = sdf(pc_ext_f)["model_out"]
    loss_out = torch.relu(1-pred_far).mean()

    loss =  10*loss_surface_eik + 1000*loss_surface_ci  + 10*loss_ext + 10*loss_surface_ne +1*loss_ext_eik + 1*loss_out
    V_loss = [loss_surface_ci.cpu().detach().numpy(),loss_ext.cpu().detach().numpy(),   loss_surface_eik.cpu().detach().numpy(),loss_ext_eik.cpu().detach().numpy() , loss_surface_ne.cpu().detach().numpy(), loss_out.cpu().detach().numpy()]
    return loss, V_loss
    
    

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Training script to learn sdf from a .ply mesh. "
    )

    parser.add_argument(
        "--seed", default=668123, type=int,
        help="Seed for the random-number generator."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the training."
    )
    parser.add_argument(
        "--mesh", default=None,
        help="Path to mesh"
    )
    parser.add_argument(
        "--save", default=None,
        help="Path to save the trained implicit representation"
    )
    parser.add_argument(
        "--batchsize", "-b", nargs='+', default=[5000, 1000, 1900], type = float, 
        help="Number of points to use per step of training."
    )
    parser.add_argument(
        "--dimension", "-dim", nargs='+', default=[6,2], type = float, 
        help="size of sampling space, first is far, second is close"
    )
    parser.add_argument(
        "--epochs", "-e", default=15000, type=int,
        help="Number of epochs of training to perform. If set to 0, fetches it"
        " from the configuration file."
    )


    parser.add_argument(
        "--name", default="",
        help="Add string at the end of the folder experience name"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    devstr = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"[WARNING] Selected device {args.device}, but CUDA is not"
              " available. Using CPU", file=sys.stderr)
        devstr = "cpu"
    device = torch.device(devstr)


    epochs = args.epochs
    batchsize = args.batchsize
    dim = args.dimension
    
    _, info = _read_ply(args.mesh, 0.)
    pc = info[:,0:3].to(device)
    normal = info[:,4:7].to(device)

    
    nsteps = epochs
    WARMUP_STEPS = nsteps // 10
    checkpoint_at = False
    if checkpoint_at:
        checkpoint_at = round(checkpoint_at * (4 * len(dataset) / batchsize))
        print(f"Checkpoints at every {checkpoint_at} training steps")
    else:
        print("Checkpoints disabled")

    print(f"Total # of training steps = {nsteps}")

    model = SIREN(3, 1, [128, 128, 128,128,128,128],
                  w0=60, delay_init=False).to(device)
    print(model)

    
    experimentpath = args.save
    
    model.zero_grad(set_to_none=True)
    lr = 10**-4
    opt = torch.optim.Adam(
        lr=lr,
        params=list(model.parameters()) 
        )


    best_loss = torch.inf
    losss = []
    early_stop = 0
    k=1
    for i in range(nsteps):
        opt.zero_grad()
        pc_surface,choice, pc_ext_c, pc_ext_f = sample_pc(pc, batchsize[0], batchsize[1], batchsize[2], device, dim[0], dim[1])
        loss, V_loss= loss_sdf(model,pc_ext_c, pc_ext_f,pc_surface,choice,pc,normal,device)
        losss.append(V_loss)
        loss.backward()
        opt.step()

        if (i+1)%100 == 0 :
            print(f"it {int(100*i/nsteps)}:  {loss.item()}")
        if loss.isnan():
            print("nan")
            print(i)
        if (i+1)%250 ==0 :
            set_opt_lr_value(opt,lr/k)
            k=k+1
            #torch.save(sdf.state_dict(),"sdf/SDF_"+shape+"_test_sin2.pth")
        if loss<0.95*best_loss and i > WARMUP_STEPS: 
            best_loss = loss
            #print("minloss : " + str(best_loss))
            torch.save(model.state_dict(), experimentpath)
            early_stop = 0
        if early_stop> 3000 : break
        early_stop +=1
