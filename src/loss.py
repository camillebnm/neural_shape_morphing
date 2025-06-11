# coding: utf-8

import torch
from torch.functional import F
from src.diff_operators import (divergence, gradient, mean_curvature,
                                 vector_dot,jacobien,laplacien_jac,mat_div)

class LossMorphing(torch.nn.Module):
    """Morphing between two neural implict functions."""
    def __init__(self, model_list, time_list, model_V = None):
        super().__init__()
        # Define the models
        self.model1 = model_list[0]
        self.model2 = model_list[1]
        self.t1 = time_list[0]
        self.t2 = time_list[1]
        self.V = model_V

    def morphing_to_NI(self, grad, sdf, coords, sdf_target, grad_target, scale=1):
        ft = grad[..., 3].unsqueeze(-1)
        grad_3d = grad[..., :3]
        grad_norm = torch.norm(grad_3d, dim=-1).unsqueeze(-1)

        # unit_grad = grad/grad_norm
        # div = divergence(unit_grad, coords)

        target_deformation = sdf_target - sdf

        # additional_weight = vector_dot(grad_3d, grad_target)

        target_deformation *= torch.exp(-sdf**2)  # gives priority to the zero-level set

        # deformation = - scale*target_deformation - 0.0005*div
        # deformation = - 0.0005*div
        deformation = - scale * target_deformation

        return (ft + deformation * grad_norm) ** 2

    def vector_field(self, coords, grad_src, grad_dst, t_src, t_dst):
        if self.V is None : 
            grad_norm_src = torch.norm(grad_src, dim=-1).unsqueeze(-1)
            grad_norm_dst = torch.norm(grad_dst, dim=-1).unsqueeze(-1)
            V_src = grad_src/grad_norm_src
            V_dst = grad_dst/grad_norm_dst
    
            # return V_src
    
            time = coords[..., 3].unsqueeze(-1)
    
            len = t_dst - t_src
            time = (time-t_src)/len
    
            V = (1-time)*V_src + time*V_dst
            return V
        else : 
            return self.V(coords)

    def level_set_equation(self, grad, sdf, coords, sdf_target, grad_source, grad_target, scale=1):
        ft = grad[..., 3].unsqueeze(-1)
        if self.V is None : 
    
            target_deformation = sdf_target - sdf
    
            # additional_weight = vector_dot(grad_3d, grad_target)
    
            target_deformation *= torch.exp(-sdf**2) #gives priority to the zero-level set
    
            deformation = -scale * target_deformation
    
            V = deformation * self.vector_field(
                coords, grad_source, grad_target, self.t1, self.t2
            )
        else : 
            V = self.V(coords, sdf=sdf)["model_out"]
        
        dot = vector_dot(grad[..., :3], V)

        return (ft + dot)**2
        
    def level_set_equation_insd(self, grad, sdf, coords, sdf_target, grad_source, grad_target, scale=1):
        n = sdf.shape[0]
        d=3
        ft = grad[..., 3].unsqueeze(-1)

        V_out = self.V(coords, sdf=sdf)
        V = V_out["model_out"]

        JV_temps = jacobien(V, V_out["model_in"])#[...,:-1]

        N_gradf = grad[...,:3]/grad[...,:3].norm(dim=-1,keepdim=True)

        L = torch.matmul(JV_temps, N_gradf.view(n,d,1)).squeeze()
        R = -torch.sum(L* N_gradf, dim=-1)
        RHS = sdf*R
        dot = vector_dot(grad[..., :3], V)

        L_V_temps = laplacien_jac(JV_temps, V_out["model_in"])
        
        div = mat_div(JV_temps)

        return (ft + dot + 1e0 * RHS)**2,  (-1e-2 * L_V_temps + 1e0 * V)**2, div**2
        
    
    def loss_insd_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # trained model1
        trained_model1 = self.model1(coords[..., 0:3])
        trained_model1_out = trained_model1['model_out']
        trained_model1_in = trained_model1['model_in']
        grad_trained_model1 = gradient(trained_model1_out, trained_model1_in)

        # trained model2
        trained_model2 = self.model2(coords[..., 0:3])
        trained_model2_out = trained_model2['model_out']
        trained_model2_in = trained_model2['model_in']
        grad_trained_model2 = gradient(trained_model2_out, trained_model2_in)

        morphing_constraint, V_constraint, div_constraint = self.level_set_equation_insd(grad, pred_sdf,
        coords, trained_model2_out, grad_trained_model1, grad_trained_model2,
        scale=1) #en pratique ici il ne faut que grad et coord mais je garde les autres au cas ou. Pourquoi pas pour réutiliser la scale. 

        # Restricting the gradient (fx, fy, fz, ft) of the SIREN function f to
        # the space: (fx, fy, fz)
        grad = grad[..., 0:3]
        time = coords[..., 3].unsqueeze(-1)

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            time == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.where(
            time == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        normal_constraint = torch.where(
            time == self.t1,
            1 - F.cosine_similarity(grad, grad_trained_model1, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        normal_constraint = torch.where(
            time == self.t2,
            (1 - F.cosine_similarity(grad, grad_trained_model2, dim=-1)[..., None]),
            normal_constraint
        )

        return {
            "sdf_constraint": sdf_constraint.mean() * 1e4,
            "normal_constraint": normal_constraint.mean() * 1e1,
            # "morphing_constraint": morphing_constraint.mean() * 1e3,
            "morphing_constraint": morphing_constraint.mean() * 1e4,
            "V_constraint" : V_constraint.mean() * 1e1,
            "div_constraint" : div_constraint.mean() * 1e1        }

    def loss_nise_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # trained model1
        trained_model1 = self.model1(coords[..., 0:3])
        trained_model1_out = trained_model1['model_out']
        trained_model1_in = trained_model1['model_in']
        grad_trained_model1 = gradient(trained_model1_out, trained_model1_in)

        # trained model2
        trained_model2 = self.model2(coords[..., 0:3])
        trained_model2_out = trained_model2['model_out']
        trained_model2_in = trained_model2['model_in']
        grad_trained_model2 = gradient(trained_model2_out, trained_model2_in)

        morphing_constraint = self.morphing_to_NI(
            grad, pred_sdf, coords, trained_model2_out, grad_trained_model2,
            scale=0.5
        )
        # morphing_constraint = self.level_set_equation(grad, pred_sdf,
        # coords, trained_model2_out, grad_trained_model1, grad_trained_model2,
        # scale=10)

        # Restricting the gradient (fx, fy, fz, ft) of the SIREN function f to
        # the space: (fx, fy, fz)
        grad = grad[..., 0:3]
        time = coords[..., 3].unsqueeze(-1)

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            time == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.where(
            time == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        normal_constraint = torch.where(
            time == self.t1,
            1 - F.cosine_similarity(grad, grad_trained_model1, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        normal_constraint = torch.where(
            time == self.t2,
            (1 - F.cosine_similarity(grad, grad_trained_model2, dim=-1)[..., None]),
            normal_constraint
        )

        return {
            "sdf_constraint": sdf_constraint.mean() * 1e4,
            "normal_constraint": normal_constraint.mean() * 1e1,
            # "morphing_constraint": morphing_constraint.mean() * 1e3,
            "morphing_constraint": morphing_constraint.mean() * 1e1,
        }

    def loss_AVSDF_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        grad = gradient(pred_sdf, coords)

        # trained model1
        trained_model1 = self.model1(coords[..., 0:3])
        trained_model1_out = trained_model1['model_out']
        trained_model1_in = trained_model1['model_in']
        grad_trained_model1 = gradient(trained_model1_out, trained_model1_in)

        # trained model2
        trained_model2 = self.model2(coords[..., 0:3])
        trained_model2_out = trained_model2['model_out']
        trained_model2_in = trained_model2['model_in']
        grad_trained_model2 = gradient(trained_model2_out, trained_model2_in)

        morphing_constraint = self.level_set_equation(grad, pred_sdf,
        coords, trained_model2_out, grad_trained_model1, grad_trained_model2,
        scale=1) #en pratique ici il ne faut que grad et coord mais je garde les autres au cas ou. Pourquoi pas pour réutiliser la scale. 

        # Restricting the gradient (fx, fy, fz, ft) of the SIREN function f to
        # the space: (fx, fy, fz)
        grad = grad[..., 0:3]
        time = coords[..., 3].unsqueeze(-1)

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            time == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.exp(-pred_sdf**2)*torch.where(
            time == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        normal_constraint = torch.where(
            time == self.t1,
            1 - F.cosine_similarity(grad, grad_trained_model1, dim=-1)[..., None],
            torch.zeros_like(grad[..., :1])
        )

        normal_constraint = torch.where(
            time == self.t2,
            (1 - F.cosine_similarity(grad, grad_trained_model2, dim=-1)[..., None]),
            normal_constraint
        )

        eikonal_constraint = torch.exp(-pred_sdf**2)*(grad.norm(dim=-1) - 1.)**2

        aa =  {
            "sdf_constraint": sdf_constraint.mean() * 1e4,
            "normal_constraint": normal_constraint.mean() * 1e1,
            # "morphing_constraint": morphing_constraint.mean() * 1e3,
            "morphing_constraint": morphing_constraint.mean() * 1e5,
            "eikonal_constraint" : eikonal_constraint.mean() * 1e0
        }
        #print(aa)
        return aa

    def loss_lipschitz_interpolation(self, X, gt):
        coords = X["model_in"]
        pred_sdf = X["model_out"]

        # trained model1
        trained_model1 = self.model1(coords[..., :3])
        trained_model1_out = trained_model1['model_out'].detach()

        # trained model2
        trained_model2 = self.model2(coords[..., :3])
        trained_model2_out = trained_model2['model_out'].detach()

        # Initial-boundary constraints of the Eikonal equation at t=0
        sdf_constraint = torch.where(
            coords[..., 3].unsqueeze(-1) == self.t1,
            (trained_model1_out - pred_sdf) ** 2,
            torch.zeros_like(pred_sdf)
        )
        sdf_constraint = torch.where(
            coords[..., 3].unsqueeze(-1) == self.t2,
            (trained_model2_out - pred_sdf) ** 2,
            sdf_constraint
        )

        return {
            "sdf_constraint": sdf_constraint.mean()*1e2,
        }

    def forward(self, X, gt, param = None ):
        if param =="lipschitz" : return self.loss_lipschitz_interpolation(X, gt)
        elif param =="FreeV" : return self.loss_AVSDF_interpolation(X, gt)
        elif param =="ADADIV" : return self.loss_AVSDF_interpolation(X, gt)
        elif param =="NULLDIV" : return self.loss_AVSDF_interpolation(X, gt)
        elif param=="nise" : return self.loss_nise_interpolation(X, gt)
        elif param=="LF-INSD" : return self.loss_insd_interpolation(X,gt)
        else  : print("NO VALID PARAMETERS FOR THE LOSS")
        
