import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from model import MLP, Transformer
from pyhessian import hessian
from util import get_data

dev = torch.device('cpu')

model = Transformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2, act_fn='wave')
model.load_state_dict(torch.load('./trans_wave/reaction_diffusion_trans_model.pkl', map_location=torch.device('cpu')))

# model = MLP(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4, act_fn='tanh')
# model.load_state_dict(torch.load('./mlp_tanh/reaction_diffusion_mlp_model.pkl', map_location=torch.device('cpu')))

res, b_left, b_right, b_upper, b_lower = get_data([0,2*np.pi], [0,1], 101, 101)

print('load done')

res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(dev)
# hessian = hessian(model=model, data=(res[:,0:1], res[:,1:2]))

# print('hessian done')

# ev, evec = hessian.eigenvalues(top_n=2)

ev = np.load('./value_trans_diffusion.npy')
evec = torch.load('./vector_trans_diffusion.pt')

print('ev done')

pev1 = np.linspace(-0.1, 0.1, 101)
pev2 = np.linspace(-0.1, 0.1, 101)

pev1_mesh, pev2_mesh = np.meshgrid(pev1,pev2)
pmat = np.concatenate((np.expand_dims(pev1_mesh, -1), np.expand_dims(pev2_mesh, -1)), axis=-1).reshape(-1,2)

print(ev[0], ev[1])

x_res, t_res = res[:,0:1], res[:,1:2]

model = Transformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2, act_fn='wave')
model.load_state_dict(torch.load('./trans_wave/reaction_diffusion_trans_model.pkl', map_location=torch.device('cpu')))

# model = MLP(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4, act_fn='tanh')
# model.load_state_dict(torch.load('./mlp_tanh/reaction_diffusion_mlp_model.pkl', map_location=torch.device('cpu')))

# model.to(dev)

# orig_dict = deepcopy(model.state_dict())

loss_track = []

for pev in tqdm(pmat):
    pev1, pev2 = pev[0], pev[1]

    model.load_state_dict(torch.load('./trans_wave/reaction_diffusion_trans_model.pkl', map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load('./mlp_tanh/reaction_diffusion_mlp_model.pkl', map_location=torch.device('cpu')))
    state_dict = model.state_dict()
    # state_dict = deepcopy(orig_dict)

    for i,key in enumerate(state_dict.keys()):
        state_dict[key] = state_dict[key] + pev1*evec[0][i] + pev2*evec[1][i]
    model.load_state_dict(state_dict)

    pred_res = model(x_res, t_res)

    u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
    # u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]

    loss_res = torch.mean((u_t - 5*u_xx - 5*pred_res*(1-pred_res)) ** 2)

    loss_track.append(loss_res.item())


loss_track = np.array(loss_track).reshape(101,101)

# np.save('./value_trans_diffusion.npy', ev)
# torch.save(evec, './vector_trans_diffusion.pt')
np.save('scape_trans_diffusion.npy', loss_track)