# PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks

## Publication

Implementation of the paper "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks."

Authors: Leo Zhiyuan Zhao, Xueying Ding, B.Aditya Prakash

Paper + Appendix: [https://arxiv.org/abs/2307.11833](https://arxiv.org/abs/2307.11833)

## Training

We also provide demo notebooks for convection, 1d_reaction, 1d_wave, and Navier-Stokes PDEs. The demos include all code for training, testing, and ground truth acquirement.

To visualize the loss landscape, run the above command to train and save the model first, then run the script:

```
python3 vis_landscape.py
```

Please adapt the model path accordingly.

## Contact

If you have any questions about the code, please contact Leo Zhiyuan Zhao at  ```leozhao1997[at]gatech[dot]edu```.

## Citation

If you find our work useful, please cite our work:

```
@article{zhao2023pinnsformer,
  title={PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks},
  author={Zhao, Leo Zhiyuan and Ding, Xueying and Prakash, B Aditya},
  journal={arXiv preprint arXiv:2307.11833},
  year={2023}
}
```
