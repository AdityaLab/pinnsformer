import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse
import random
import os

from model.fls import FLS
from model.pinn import PINNs
from model.pinnsformer import PINNsformer
from model.qres import QRes  

from util import *

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def train(res, b_left, b_right, b_upper, b_lower, eq_name, eq_param, device, model='mlp', epoch=1000, step=5, stepsize=1e-4):
    if model == 'mlp':
        model = MLP(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    if model == 'trans':
        model = Transformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)
    if model == 'qres':
        model = QRes(in_dim=2, hidden_dim=512, out_dim=1, num_layer=2).to(device)
    if model == 'fls':
        model = FLS(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    
    model.apply(init_weights)

    optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

    print(model)
    print(get_n_params(model))

    if model == 'trans':
        res = make_time_sequence(res, num_step=step, step=stepsize)
        b_left = make_time_sequence(b_left, num_step=step, step=stepsize)
        b_right = make_time_sequence(b_right, num_step=step, step=stepsize)
        b_upper = make_time_sequence(b_upper, num_step=step, step=stepsize)
        b_lower = make_time_sequence(b_lower, num_step=step, step=stepsize)

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
    b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
    b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
    b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

    if model == 'trans':
        x_res, t_res = res[:,:,0:1], res[:,:,1:2]
        x_left, t_left = b_left[:,:,0:1], b_left[:,:,1:2]
        x_right, t_right = b_right[:,:,0:1], b_right[:,:,1:2]
        x_upper, t_upper = b_upper[:,:,0:1], b_upper[:,:,1:2]
        x_lower, t_lower = b_lower[:,:,0:1], b_lower[:,:,1:2]
        
    else:
        x_res, t_res = res[:,0:1], res[:,1:2]
        x_left, t_left = b_left[:,0:1], b_left[:,1:2]
        x_right, t_right = b_right[:,0:1], b_right[:,1:2]
        x_upper, t_upper = b_upper[:,0:1], b_upper[:,1:2]
        x_lower, t_lower = b_lower[:,0:1], b_lower[:,1:2]

    loss_track = []
    loss_track = []

    for i in tqdm(range(epoch)):
        def closure():
            pred_res = model(x_res, t_res)
            pred_left = model(x_left, t_left)
            pred_right = model(x_right, t_right)
            pred_upper = model(x_upper, t_upper)
            pred_lower = model(x_lower, t_lower)

            u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
            u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]

            if eq_name == 'burger':
                loss_res = torch.mean((u_t + pred_res * u_x - (eq_param['c']/torch.pi)*u_xx) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean((pred_left[:,0] + torch.sin(torch.pi * x_left[:,0])) ** 2)

            if eq_name == 'convection':
                loss_res = torch.mean((u_t + eq_param['beta'] * u_x) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean((pred_left[:,0] - torch.sin(x_left[:,0])) ** 2)

            if eq_name == '1d_reaction':
                loss_res = torch.mean((u_t - eq_param['p'] * pred_res * (1-pred_res)) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean((pred_left[:,0] - torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))) ** 2)

            if eq_name == 'reaction_diffusion':
                loss_res = torch.mean((u_t - eq_param['v']*u_xx - eq_param['p']*pred_res*(1-pred_res)) ** 2)
                loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
                loss_ic = torch.mean((pred_left[:,0] - torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))) ** 2)

            if eq_name == 'helmholtz':
                pi = torch.tensor(np.pi)

                u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), retain_graph=True, create_graph=True)[0]
                loss_res = torch.mean((
                    u_xx + u_tt + (eq_param['k']**2) * pred_res \
                        + ((pi * eq_param['a1'])**2 + (pi * eq_param['a2'])**2 - eq_param['k']**2) \
                        * torch.sin(pi * eq_param['a1'] * x_res) * torch.sin(pi * eq_param['a2'] * t_res) 
                )**2)
                loss_bc = torch.mean((pred_upper) **2) + torch.mean((pred_lower) **2)
                loss_ic = torch.mean((pred_left[:,0]) **2) + torch.mean((pred_right[:,0]) **2)


            loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

            loss = loss_res + loss_bc + loss_ic
            optim.zero_grad()
            loss.backward()
            return loss

        optim.step(closure)

        if i%100 == 0:
            print('EPOCH: {:d}, Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(i, loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))


    return model, np.array(loss_track)


def test(res, model, device, step=5, stepsize=1e-4):
    model.to(device)

    res = make_time_sequence(res, num_step=step, step=stepsize)

    res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
    x_res, t_res = res[:,:,0:1], res[:,:,1:2]

    with torch.no_grad():
        pred = model(x_res, t_res)[:,0:1]
        pred = pred.cpu().detach().numpy()

    return pred


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='PyTorch PINNsformer')

    parser.add_argument('--model', type=str, default='trans', help='select model by name')
    parser.add_argument('--save_pred', type=bool, default=True, help='save prediction')
    parser.add_argument('--save_loss', type=bool, default=True, help='save training loss')
    parser.add_argument('--save_model', type=bool, default=True, help='save model')

    parser.add_argument('--eq_name', type=str, default='convection', help='equation name')

    parser.add_argument('--step', type=int, default=5, help='length of pseudo sequence')
    parser.add_argument('--stepsize', type=float, default=1e-4, help='stepsize of pseudo sequence')

    parser.add_argument('--dev', type=str, default='cuda:0', help='device name')   
    args = parser.parse_args()

    device = args.dev

    if args.eq_name == 'convection':
        eq_set = {'eq_name':'convection', 'eq_param':{'beta':50}, 'x_range':[0,2*np.pi], 'y_range':[0,1]}
    if args.eq_name == '1d_reaction':
        eq_set = {'eq_name':'1d_reaction', 'eq_param':{'p':5}, 'x_range':[0,2*np.pi], 'y_range':[0,1]}
    if args.eq_name == 'reaction_diffusion':
        eq_set = {'eq_name':'reaction_diffusion', 'eq_param':{'p':5, 'v':5}, 'x_range':[0,2*np.pi], 'y_range':[0,1]}
    if args.eq_name == 'burger':
        eq_set = {'eq_name':'burger', 'eq_param':{'c':0.01}, 'x_range':[-1,1], 'y_range':[0,1]}
    if args.eq_name == 'helmholtz':
        eq_set = {'eq_name':'helmholtz', 'eq_param':{'a1':1, 'a2':4, 'k':1}, 'x_range':[-1,1], 'y_range':[-1,1]}

    res, b_left, b_right, b_upper, b_lower = get_data(eq_set['x_range'], eq_set['y_range'], 51, 51)
    model, loss_track = train(res, b_left, b_right, b_upper, b_lower, eq_set['eq_name'], eq_set['eq_param'], device=device, model=args.model, step=args.step, stepsize=args.stepsize)
    res_test, _, _, _, _ = get_data(eq_set['x_range'], eq_set['y_range'], 101, 101)
    pred = test(res_test, model, device)

    if not os.path.exists('./{}'.format(args.model)):
        os.mkdir('./{}'.format(args.model))

    if args.save_pred:
        np.save('./{}/{}_{}_pred.npy'.format(args.model, eq_set['eq_name'], args.model), pred)

    if args.save_loss:
        np.save('./{}/{}_{}_loss.npy'.format(args.model, eq_set['eq_name'], args.model), loss_track)

    if args.save_model:
        torch.save(model.state_dict(), './{}/{}_{}_model.pkl'.format(args.model, eq_set['eq_name'], args.model))