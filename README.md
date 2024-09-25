**Notes**:

- Experiments with Neural ODE style methods for white-light images

- Uses `torchdiffeq`

**Pending**:

- PNODEs on larger CME dataset with correct use of dataloaders

- Modified PNODEs to handle constraints such as continuity, etc.

- Training on segmented instead of raw white light images

- Setups for pushing observation images through network. Latent space loss, paired autoencoders, etc.

- Training Continuous Conv architectures for prediction of arrival time and other quantities of interest at 1au


Miscellaneous:

- Sparse CNN: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf

- NeuroDiffEq - https://github.com/NeuroDiffGym/neurodiffeq - general ideas are very different, somewhat closer to PINNs?

- Some analytical CFD examples in section 3 of this study? [RG Link](https://www.researchgate.net/publication/338867908_Artificial_Neutral_Networks_ANNs_Applied_as_CFD_Optimization_Techniques)

- torch metrics like MSE, absolute loss? some experiments: [Gist](https://gist.github.com/aniketjivani/231d63efee9308d8fc1fee7d1cd61bb6)


For code on ME-Maverick (mostly to remain untouched):

Uses virtual environment ptvenv
From home directory, do `source ptvenv/bin/activate`
And then launch Jupyter notebook from virtual env