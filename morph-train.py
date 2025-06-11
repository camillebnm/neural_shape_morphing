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
from torch.utils.tensorboard import SummaryWriter
import yaml
from src.dataset import SpaceTimePointCloudNI
from src.loss import LossMorphing
from src.meshing import create_mesh, save_ply
from src.model import SIREN, HODGE_SIREN, LipschitzMLP, DIV_SIREN
from src.util import create_output_paths, scale_volume, eval_mass_sdf, scale
from src.diff_operators import vector_dot

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Default training script when using Neural Implicits for"
        " SDF querying and mean curvature experiments. Note that command line"
        " arguments have precedence over configuration file values."
    )
    parser.add_argument(
        "experiment_config", type=str, help="Path to the YAML experiment"
        " configuration file."
    )
    parser.add_argument(
        "--initial_condition", "-i", action="store_true", default=False,
        help="Initialization method for the model. If set, uses the first"
        " initial condition for the morphing network weigths. By default,"
        " uses SIREN's method."
    )
    parser.add_argument(
        "--seed", default=668123, type=int,
        help="Seed for the random-number generator."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the training."
    )
    parser.add_argument(
        "--batchsize", "-b", default=0, type=int,
        help="Number of points to use per step of training. If set to 0,"
        " fetches it from the configuration file."
    )
    parser.add_argument(
        "--epochs", "-e", default=0, type=int,
        help="Number of epochs of training to perform. If set to 0, fetches it"
        " from the configuration file."
    )
    parser.add_argument(
        "--time_benchmark", "-t", action="store_true", help="Indicates that we"
        " are running a training time measurement. Disables writing to"
        " tensorboard, model checkpoints, best model serialization and mesh"
        " generation during training."
    )
    parser.add_argument(
        "--kaolin", action="store_true", default=False, help="When saving"
        " mesh checkpoints, use kaolin format, or simply save the PLY files"
        " (default). Note that this respects the checkpoint configuration in"
        " the experiment files, if no checkpoints are enabled, then nothing"
        " will be saved."
    )
    parser.add_argument(
        "--param", "-p", default=None,choices = ["ADADIV","NULLDIV", "FreeV", "nise", "LF-INSD"], 
        help="Type of training to be performed, none for nise, lispschitz for lipmlp"
        "NULLDIV,ADADIV,FreeV for our own baseline and LF-INSD for our implementation of INSD"
    )
    parser.add_argument(
        "--name", default="",
        help="Add string at the end of the folder experience name"
    )
    parser.add_argument(
        "--scale", default=0,type=float, 
        help="reduce the smallest shape by a factor alpha"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(args.experiment_config, 'r') as f:
        config = yaml.safe_load(f)

    devstr = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"[WARNING] Selected device {args.device}, but CUDA is not"
              " available. Using CPU", file=sys.stderr)
        devstr = "cpu"
    device = torch.device(devstr)

    training_config = config["training"]
    training_data_config = config["training_data"]
    training_mesh_config = training_data_config["mesh"]

    epochs = training_config.get("n_epochs", 100)
    if args.epochs:
        epochs = args.epochs

    batchsize = training_data_config.get("batchsize", 20000)
    if args.batchsize:
        batchsize = args.batchsize

    meshdata = []
    ictimes = []
    for meshpath, data in training_mesh_config.items():
        ictimes.append(data['t'])
        meshdata.append((meshpath, data["ni"], data['t'], data["omega_0"]))

    dataset = SpaceTimePointCloudNI(meshdata, batchsize, size_domain = training_data_config["size"])

    nsteps = round(epochs * (4 * len(dataset) / batchsize))
    WARMUP_STEPS = nsteps // 10
    checkpoint_at = training_config.get("checkpoints_at_every_epoch", 0)
    if checkpoint_at:
        checkpoint_at = round(checkpoint_at * (4 * len(dataset) / batchsize))
        print(f"Checkpoints at every {checkpoint_at} training steps")
    else:
        print("Checkpoints disabled")

    print(f"Total # of training steps = {nsteps}")

    network_config = config["network"]

    #Computing of the value needed for scaling
    pc_source = dataset.vertices_ni[0][0][: , :3]
    pc_cible = dataset.vertices_ni[1][0][ :, :3]
    sdf0 = dataset.vertices_ni[0][1]
    sdf1 = dataset.vertices_ni[1][1]
    m1 = eval_mass_sdf(sdf0, size = 1, n=250, device=device) 
    m2 = eval_mass_sdf(sdf1, size = 1, n=250, device=device) 
    print("Checking correctness of the shape loaded : ")
    print(f" Volume 1: {m1}, Volume 2 : {m2}")
    a1 = abs(dataset.vertices_ni[0][1](dataset.vertices_ni[0][0][ :1000, :3])["model_out"]).mean()
    a2 = abs(dataset.vertices_ni[1][1](dataset.vertices_ni[1][0][ :1000, :3])["model_out"]).mean()
    print(f"Diff Volume : {m1-m2}")
    print(f"approx shape 1 : {a1}, approx shape 2 : {a2}", flush=True)
    
    #Manual scaling if wanted
    if args.scale : 
        print( f" Scaling by user of {args.scale}")
        if m1<m2 : 
            dataset.vertices_ni[0][0][ :, :3] = args.scale*dataset.vertices_ni[0][0][ :, :3]
            scale(dataset.vertices_ni[0][1],args.scale)
        else : 
            dataset.vertices_ni[1][0][ :, :3] = args.scale*dataset.vertices_ni[1][0][ :, :3]
            scale(dataset.vertices_ni[1][1], args.scale)
    
    #Perform automatic scaling of the shape if you use a method based on volume conservation. 
    if args.param =="ADADIV" or args.param =="LF-INSD" or args.param =="NULLDIV": 
        #Scaling
        print("Scaling shapes ...")
        alpha, first = scale_volume(sdf0, sdf1, 10**-3)
        #The volume scaling is based on an iterative method, the last parameter is the error threshold of the differences between the volume of the 2 shapes
        print(f"alpha : {alpha}")
        if first : 
            dataset.vertices_ni[0][0][ :, :3] = alpha*dataset.vertices_ni[0][0][ :, :3]

        else : 
            dataset.vertices_ni[1][0][ :, :3] = alpha*dataset.vertices_ni[1][0][ :, :3]

    
        
    #verification :
    print("Post scaling transformation verification : ")
    m1 = eval_mass_sdf(sdf0, size = 1, n=250, device=device) 
    m2 = eval_mass_sdf(sdf1, size = 1, n=250, device=device) 
    a1 = abs(dataset.vertices_ni[0][1](dataset.vertices_ni[0][0][ :1000, :3])["model_out"]).mean()
    a2 = abs(dataset.vertices_ni[1][1](dataset.vertices_ni[1][0][ :1000, :3])["model_out"]).mean()
    print(f" Volume 1: {m1}, Volume 2 : {m2}")
    print(f"Diff vplume : {m1-m2}")
    print(f"approx shape 1 : {a1}, approx shape 2 : {a2}", flush=True)
    #The print above allow to verify that the scalling went well. 
    
    #Init of the implicit surface model
    if args.param == "lipschitz" : 
        model = LipschitzMLP(4, 1, network_config["hidden_layer_nodes"], w0 =network_config["omega_0"] ).to(device) 
    else : 
        model = SIREN(4, 1, network_config["hidden_layer_nodes"],
                  w0=network_config["omega_0"], delay_init=True).to(device)
    print(model)

    name = args.param
    
        
    experiment = osp.split(args.experiment_config)[-1].split('.')
    experimentpath = create_output_paths(
        "results",
        experiment[0],
        overwrite=False,
        name = name + args.name

    )
    #Warning, your args.experiment_config cannot contain more than one ".", for exemple, giving the experiement name "test.trial.yaml" will create an output folder named "test/"
    
    writer = SummaryWriter(osp.join(experimentpath, 'summaries'))

    model.zero_grad(set_to_none=True)


    init_method = network_config.get("init_method", "siren")
    if not (args.param=="lipschitz"):
        model.reset_weights() 
        if args.initial_condition:
            init_method = "initial_condition"

        if init_method =="SIREN": 
            print("Performing SIREN initialization")
            model.reset_weights()
        if init_method == "initial_condition":
            print("Performing NISE initialisation")
            w0 = model.w0
            model.update_omegas(15)
            model.from_pretrained_initial_condition(torch.load(meshdata[0][1]))
            model.update_omegas(w0)

        if "timesampler" in training_mesh_config:
            timerange = training_mesh_config["timesampler"].get(
                "range", [-1.0, 1.0]
            )
            dataset.time_sampler = torch.distributions.uniform.Uniform(
                timerange[0], timerange[1]
            )
        

    print("###############################")
    
    model_V = None
    if args.param == "FreeV" or args.param == "LF-INSD" :  
        model_V = SIREN(4, 3, network_config["hidden_layer_nodes"],
                      w0=network_config["omega_0"], delay_init=False).to(device) #même dim que le modele de la LSE
    if args.param == "ADADIV" : 
        model_V = HODGE_SIREN(device, 4, 4, network_config["hidden_layer_nodes"],
                      w0=network_config["omega_0"], delay_init=False).to(device) 
    if args.param =="NULLDIV" :
        model_V = DIV_SIREN(device, 4, 3, network_config["hidden_layer_nodes"],
                      w0=network_config["omega_0"], delay_init=False).to(device) 
    print(model_V)  

    if args.param =="nise" or args.param == "lipschitz":
        optim = torch.optim.Adam(
        lr=1e-4,
        params=list(model.parameters()) 
        )
    else : 
        optim = torch.optim.Adam(
            lr=5e-5,
            params=list(model.parameters())+list(model_V.parameters()) 
            #One optimizer for both model
        )

    trainingpts = torch.zeros((batchsize, 4), device=device)
    trainingnormals = torch.zeros((batchsize, 3), device=device)
    trainingsdf = torch.zeros((batchsize), device=device)

    n_on_surface = training_data_config.get("n_on_surface", math.floor(batchsize * 0.25))
    n_off_surface = training_data_config.get("n_off_surface", math.ceil(batchsize * 0.25))
    n_int_times = training_data_config.get("n_int_times", batchsize - (n_on_surface + n_off_surface))

    allni = [vertni[1] for vertni in dataset.vertices_ni]
    lossmorph = LossMorphing(allni, ictimes, model_V)

    checkpoint_times = training_config.get("checkpoint_times", ictimes)

    updated_config = copy.deepcopy(config)
    updated_config["network"]["init_method"] = "siren"
    updated_config["training"]["n_epochs"] = epochs
    updated_config["training_data"]["batchsize"] = batchsize
    updated_config["training_data"]["n_on_surface"] = n_on_surface
    updated_config["training_data"]["n_off_surface"] = n_off_surface
    updated_config["training_data"]["n_int_times"] = n_int_times

    with open(osp.join(experimentpath, "config.yaml"), 'w') as f:
        yaml.dump(updated_config, f)
    best_loss = torch.inf
    best_weights = None
    omegas = dict()  # {3: 10}  # Setting the omega_0 value of t (coord. 3) to 10
    training_loss = {}

    if not KAOLIN_AVAILABLE and args.kaolin:
        print("Kaolin was selected but is not available. Switching to the"
              " usual checkpoint saving.")

    if args.kaolin and KAOLIN_AVAILABLE and not args.time_benchmark:
        timelapse = kaolin.visualize.Timelapse(
            osp.join(experimentpath, "kaolin")
        )

    
    start_training_time = time.time()

    

    for e in range(nsteps):

        data = dataset[e]
        # ===============================================================
        trainingpts[:n_on_surface, ...] = data["on_surf"][0]
        trainingnormals[:n_on_surface, ...] = data["on_surf"][1]
        trainingsdf[:n_on_surface] = data["on_surf"][2]

        trainingpts[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][0]

        trainingnormals[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][1]
        trainingsdf[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][2].squeeze()

        trainingpts[(n_on_surface + n_off_surface):, ...] = data["int_times"][0]
        trainingnormals[(n_on_surface + n_off_surface):, ...] = data["int_times"][1]
        trainingsdf[(n_on_surface + n_off_surface):, ...] = data["int_times"][2]
        

        gt = {
            "sdf": trainingsdf.float().unsqueeze(1),
            "normals": trainingnormals.float(),
        }
        optim.zero_grad(set_to_none=True)
        if args.param =="lipschitz": y = model(trainingpts)
        else : y = model(trainingpts, omegas=omegas)
        loss = lossmorph(y, gt, param = args.param)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
            if not args.time_benchmark:
                writer.add_scalar(f"train/{k}_term", v.detach().item(), e)
            if k not in training_loss:
                training_loss[k] = [v.detach().item()]
            else:
                training_loss[k].append(v.detach().item())


        running_loss.backward()
        optim.step()

        if e > WARMUP_STEPS and best_loss > running_loss.item(): #Ne faire la sauvegarde que si le gain est significatif
            best_weights = copy.deepcopy(model.state_dict())
            best_loss = running_loss.item()
            if model_V is not None : best_weights_V  = copy.deepcopy(model_V.state_dict())
        writer.add_scalar("train/loss", running_loss.detach().item(), e)
        if not args.time_benchmark:
            writer.add_scalar("train/loss", running_loss.detach().item(), e)

            if checkpoint_at and e and not e % checkpoint_at:
                for i, t in enumerate(checkpoint_times):
                    verts, faces, normals, _ = create_mesh(
                        model,
                        t=t,
                        N=256,
                        device=device
                    )
                    if KAOLIN_AVAILABLE and args.kaolin:
                        timelapse.add_mesh_batch(
                            category=f"check_{i}",
                            iteration=e // checkpoint_at,
                            faces_list=[torch.from_numpy(faces.copy())],
                            vertices_list=[torch.from_numpy(verts.copy())]
                        )
                    else:
                        meshpath = osp.join(
                            experimentpath, "reconstructions", f"check_{e}"
                        )
                        os.makedirs(meshpath, exist_ok=True)
                        save_ply(
                            verts, faces, osp.join(meshpath, f"time_{t}.ply")
                        )

                model = model.train()

            if not e % 1000 and e > 0:
                print(f"Step {int(100*e/nsteps)}% --- Loss {running_loss.item()}", flush = True)

    training_time = time.time() - start_training_time
    print(f"training took {training_time} s")
    writer.flush()
    writer.close()

    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "weights.pth")
    )
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1)
    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "best.pth")
    )

    if model_V is not None : 
        torch.save(
            model_V.state_dict(), osp.join(experimentpath, "models", "weights_V.pth")
        )
        
        #model_V.load_state_dict(best_weights_V)
        #model.update_omegas(w0=1)
        torch.save(
            best_weights_V, osp.join(experimentpath, "models", "best_V.pth")
        )

