# coding: utf-8

import torch
from torch.autograd import grad


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    """Gradient of `y` with respect to `x`
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    return grad

#Compute the jacobien of a vector field R3xR -> R3
#The output has shape nx3x3
def jacobien(V_out,input):
	GX = gradient(V_out[:,0],input).view(input.shape[0],1,input.shape[1])
	GY = gradient(V_out[:,1],input).view(input.shape[0],1,input.shape[1])
	GZ = gradient(V_out[:,2],input).view(input.shape[0],1,input.shape[1])
	DV = torch.cat((GX,GY,GZ),axis=1)[:,:,:-1]
	return DV
#Compute the divergence of a vector from its jacobian
def mat_div(jac):
	div = []
	for mat in jac:
		div.append(torch.trace(mat))
	return torch.tensor(div)

#Compute the curl of a vector field from its jacobian
def rotationnel(J,device):
  P = torch.tensor([[0.,1,0],[0,0,1],[1,0,0]],device=device)
  JJ1 = torch.matmul(J,P)
  JJ2 = torch.matmul(torch.swapaxes(J,1,2),P)
  d = torch.diagonal(JJ1-JJ2,dim1=1,dim2=2)
  s = torch.matmul(d,P)
  return s

def vector_dot(u, v):
    return torch.sum(u * v, dim=-1, keepdim=True)


def mean_curvature(grad, x):
    grad = grad[..., 0:3]
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad/grad_norm.unsqueeze(-1)

    Km = divergence(unit_grad, x)
    return Km

#Compute the vector laplacian from the jacobian of a vector field. 
def laplacien_jac(J,x):
    Lx = divergence(J[:,0,:],x)
    Ly = divergence(J[:,1,:],x)
    Lz = divergence(J[:,2,:],x)
    DV = torch.cat((Lx,Ly,Lz),axis=1)
    return DV
