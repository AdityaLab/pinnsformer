import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from model.pinn import PINNs
from model.pinnsformer import PINNsformer
from pyhessian import hessian
from util import get_data

dev = torch.device('cpu')

# TODO: define and load the trained model
# model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)
# model.load_state_dict(torch.load('./trans_wave/reaction_diffusion_trans_model.pkl', map_location=torch.device('cpu')))

res, b_left, b_right, b_upper, b_lower = get_data([0,2*np.pi], [0,1], 101, 101)

print('load done')

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(dev)

# TODO: Compute hessian and save
# For efficiency purpose, please save and avoid recompute
# hessian = hessian(model=model, data=(res[:,0:1], res[:,1:2]))
# print('hessian done')
# ev, evec = hessian.eigenvalues(top_n=2)

# TODO: Load hessasin, evalue, evector
# ev = np.load()
# evec = torch.load()
# print('ev done')

# TODO: Perturbation range
pev1 = np.linspace(-0.1, 0.1, 101)
pev2 = np.linspace(-0.1, 0.1, 101)

pev1_mesh, pev2_mesh = np.meshgrid(pev1,pev2)
pmat = np.concatenate((np.expand_dims(pev1_mesh, -1), np.expand_dims(pev2_mesh, -1)), axis=-1).reshape(-1,2)

print(ev[0], ev[1])

x_res, t_res = res[:,0:1], res[:,1:2]

loss_track = []

for pev in tqdm(pmat):
    pev1, pev2 = pev[0], pev[1]

    # TODO: Reload the model
    # model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)
    # model.load_state_dict(torch.load('./trans_wave/reaction_diffusion_trans_model.pkl', map_location=torch.device('cpu')))
    state_dict = model.state_dict()

    # Compute model params with perturbation
    for i,key in enumerate(state_dict.keys()):
        state_dict[key] = state_dict[key] + pev1*evec[0][i] + pev2*evec[1][i]
    model.load_state_dict(state_dict)

    # Compute the PINNs loss
    # pred_res = model(x_res, t_res)
    # i.e. u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    # loss = torch.mean(u_x ** 2)
    
    loss_track.append(loss_res.item())


loss_track = np.array(loss_track).reshape(101,101)
# np.save('your_loss_landspace.npy', loss_track)