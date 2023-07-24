import torch
import torch.nn as nn
import copy


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x) + torch.cos(x)
    

class SinAct(nn.Module):
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, act, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            act,
            nn.Linear(d_ff, d_ff),
            act,
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, act):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model, act)
        self.act1 = act
        self.act2 = act
        
    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2,x2,x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, act):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model, act)
        self.act1 = act
        self.act2 = act

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x
    

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, act):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, act), N)
        self.layers = nn.ModuleList(self.layers)
        self.act = act

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)
    

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, act):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads, act), N)
        self.act = act
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)
    


class Transformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, act_fn='sin'):
        super(Transformer, self).__init__()
        if act_fn == 'sin':
            act = SinAct()
        elif act_fn == 'wave':
            act = WaveAct()
        elif act_fn == 'tanh':
            act = nn.Tanh()
        elif act_fn == 'relu':
            act = nn.ReLU()
        else:
            raise Exception('Please use a valid activation function')
        
        # self.linear_emb = nn.Sequential(*[
        #     nn.Linear(2, d_hidden),
        #     act,
        #     nn.Linear(d_hidden, d_hidden),
        #     act,
        #     nn.Linear(d_hidden, d_model)
        # ])
        self.linear_emb = nn.Linear(2, d_model)

        self.encoder = Encoder(d_model, N, heads, act)
        self.decoder = Decoder(d_model, N, heads, act)
        self.linear_out = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            act,
            nn.Linear(d_hidden, d_hidden),
            act,
            nn.Linear(d_hidden, d_out)
        ])


    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        src = self.linear_emb(src)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        output = self.linear_out(d_output)
        return output
    

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, act_fn='sin'):
        super(MLP, self).__init__()
        if act_fn == 'sin':
            act = SinAct()
        elif act_fn == 'wave':
            act = WaveAct()
        elif act_fn == 'tanh':
            act = nn.Tanh()
        elif act_fn == 'relu':
            act = nn.ReLU()
        else:
            raise Exception('Please use a valid activation function')

        layers = []
        for i in range(num_layer-1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(act)
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(act)

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, t):
        src = torch.cat((x,t), dim=-1)
        return self.linear(src)