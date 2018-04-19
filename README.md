# deformable-convolution-Neural-Network
A good example of deformable convolutional network for mnist classification


Paper Reference: Deformabel convolutional netwokrs -- Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, et al. 5 Jun 2017


Code Reference: https://github.com/felixlaumon/deform-conv

The code is written by keras with tensorflow backend. (Theano backend does not work)


Deformable convolutional networks can reduce the effect of geometric transformation on image classification accuracy. In which,  the additional convolutional layers are used to learn unknown affine transformations. On the other hand, this structure increases training cost -- time consumption and computational source.


A review of bilinear interpolation is also presented in this repository and I also explian how to realize this algorithm with tensorflow code library. ()
