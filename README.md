[![License](https://img.shields.io/github/license/analysiscenter/pydens.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)

# Learnable Hollow Kernels for Anatomical Segmentation

This is a DL environment for developing and comparing results of certain hollow organs segmentation, such as the bladder, which is especially hard to automate due to their complex geometry, vague intensity gradients in the soft tissues, and a tedious manual process of the data annotation routine. Yet, accurate localization of the walls and the cancer regions in the radiologic images of such organs is an essential step in oncology. We proposed a new class of hollow kernels that learn to 'mimic' the contours of the segmented organ, effectively replicating its shape and structural complexity. We train a series of the U-Net-like neural networks using the proposed kernels and demonstrate the superiority of the idea in various spatio-temporal convolution scenarios. Specifically, the dilated hollow-kernel architecture outperforms state-of-the-art \emph{spatial} segmentation models, whereas addition of temporal blocks with, e.g., Bi-LSTM, establishes a new multi-class baseline for the bladder segmentation challenge.

<p align="center">
<img src="/imgs/kernels.png" alt>

</p>
<p align="center">
<em>(Top:)The bladder in 3D and the hollow kernels which mimic the shape of the organ.</em>
<em>(Bottom:)Intuition behind the hollow kernels: convolution with them emphasizes the contours of the organ along with the tissue borders in the vicinity. The effect depends on the scale of the convolution kernel.</em>
</p>


### Installation as a project repository:

```
git clone https://github.com/cviaai/LEARNABLE-HOLLOW-KERNELS.git
```
In this case, you may need to manually install the dependencies.

### Important notes:

## Citing 
If you use this package in your publications or in other work, please cite it as follows:
```
@misc{Lazareva-LHK2020-git,
  author = {Elizaveta Lazareva and Oleg Y. Rogov and Olga Shegai and Denis Larionov and Dmitry V. Dylov},
  title = {Learnable Hollow Kernels for Anatomical Segmentation},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cviaai/LEARNABLE-HOLLOW-KERNELS/}}
}
```
Read more in the paper:
```
@misc{Lazareva-LHK2020,
    title={Learnable Hollow Kernels for Anatomical Segmentation,
    author={Elizaveta Lazareva and Oleg Y. Rogov and Olga Shegai and Denis Larionov and Dmitry V. Dylov},
    year={2020},
    eprint={2002.10948},
    archivePrefix={arXiv},
    primaryClass={q-bio.NC}
}
```
## Repository maintainters
Elizaveta Lazareva (main contributor)

Oleg Rogov
