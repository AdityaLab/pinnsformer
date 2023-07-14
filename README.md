# PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks

## Publication

Implementation of the paper "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks."

Authors: Leo Zhiyuan Zhao, Xueying Ding, B.Aditya Prakash

Paper + Appendix: 

## Training

To train the baseline physics-informed neural networks (PINNs):

```
python3 main.py --model mlp --eq_name convection --act tanh --dev cuda:0
```

To train the PINNsFormer:

```
python3 main.py --model trans --eq_name convection --act wave --dev cuda:0
```

The type of equation can be selected from ```{burger, convection, 1d_reaction, reaction_diffusion, helmholtz}```

To visualize the loss landscape, run the above command to train and save the model first, then run the script:

```
python3 vis_landscape.py
```

Please adapt the model path accordingly.

## Contact

If you have any questions about the code, please contact Leo Zhiyuan Zhao at  ```leozhao1997[at]gatech[dot]edu```.
