# ese546 Principles of Deep Learning project

We compared Transformer and typical autoregressive convolutional models in image generation to see whether self-attention architecture can outperform previous methods qualitatively and quantitatively. By restricting receptive field to local neighbourhoods, our model can achieve comparable performance but ends up using significantly less parameters on CIFAR-10. We also tested model capability through image completion and derive reasonable generated image conditioned on top-half part.  \
Report is available at https://github.com/lyuheng/546project/blob/main/demo/546report.pdf

### Quantitative Results <br />

|  Model Type | Params  | Bits/dim  | 
|:---:|:---:|:---:|
|  PixelCNN++ | -- |  3.09 | 
|  1D TF | block_length=256   | 3.16  |
|  2D TF | kernel_size = 4  | 3.28  | 
|  2D TF | kernel_size = 6 | 3.23  | 

### Qualitative Results 
* PixelCNN++ <br />
<img src="https://github.com/lyuheng/546project/blob/main/demo/pixelcnn_half_gen.png" width="380" height="300" />

* 1D Transformer <br />
<img src="https://github.com/lyuheng/546project/blob/main/demo/trans_half_gen.png" width="380" height="300" />

* 2D Transformer <br />
<img src="https://github.com/lyuheng/546project/blob/main/demo/trans_2d_half_gen.png" width="380" height="300" />
