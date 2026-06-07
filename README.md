# Computer Vision Final Project

## 1. Environment & Requirements

| Item               | Details                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------|
| **OS / Kernel**    | Ubuntu 22.04.3 LTS                                                                                       |
| **GPU**            | NVIDIA A100-SXM4-40GB                                                                                    |
| **Python**         | 3.10.14 (**Conda**)                                                                                      |
| **Core Libraries** | torch (2.10.0) · torchvision (0.25.0) · numpy · timm · pillow · matplotlib · segmentation_models_pytorch |


## 2. How to run

### 2-1. To download MINC & Opensurface dataset

Download from **[Dataset Download](https://drive.google.com/drive/folders/1BJFKH0cRnT5ZZMy1MzJZQfKx1G0y6-C0?usp=sharing)**.

To extract tar files will be performed automatically by **install.sh**.

### 2-2. To run
```sh
# install.sh will automatically extract tar files, set up the environment, and install necessary libraries.
source install.sh

# With checking reproducibility
# In this case, model trains three times.
source run.sh true

# Without checking reproducibility
# In this case, model trains once.
source run.sh false

# To run baseline experiment --- We trained baseline models from scratch.
source baseline.sh
```

## 3. Reproducibility scope

For the reproducibility, we added in **main.py** and **baseline.py** as follows:
```py
seed = args.seed

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True, warn_only = True)

timm.layers.set_fused_attn(False)
```

## 4. AI tools used
<!-- Edit here -->
We used Claude Code on Visual Studio Code. We used this AI tool to verify logics, to solve error, and to receive suggestions.
And to convert python2 code (process_opensurfaces_release_0.py) into python3 code.
And to create plot.py and image.py.

For implementation support we used Claude Code (Anthropic) running inside
Visual Studio Code. All design decisions were made by the human authors, who
reviewed every change and ran and validated all experiments; the AI was used
to draft code, refactor existing code, debug, and explain.

The AI was applied in the following places. It helped convert the original
CIFAR-10 classifier into a per-pixel material-segmentation pipeline for the
MINC dataset, which mainly affected main.py, encoder.py, and preprocessing.py.
For the model architecture, it was used to draft and refactor the dual-encoder
(a pretrained timm ViT-S together with a torchvision ResNet34), the
mixture-of-experts modules, and the segmentation decoder in encoder.py and
transformer.py. On the data side, it helped implement collate_fn and the MINC
mask handling in preprocessing.py, and it converted
process_opensurfaces_release_0.py from Python 2 to Python 3.

We also used it for debugging and diagnosis: resolving NaN loss, a CUDA/NVML
driver-version mismatch, and a non-deterministic nll_loss2d warning, and
identifying that our initially low mIoU was caused by class collapse from
severe pixel imbalance. Based on that diagnosis it suggested the fixes we
adopted, namely pretrained encoders, data augmentation, automatic mixed
precision, a differential learning rate, and best-validation checkpointing.
Finally, it helped implement the reproducibility settings (seeding and
deterministic algorithms) in main.py, set up the segmentation_models_pytorch
baseline sweep in baseline.py and run_3.sh, write the plotting and
visualization utilities plot.py and image.py, and draft parts of this README.

## 5. Baseline sources

<!-- Version 1 -->
| Baseline Model         | Paper                                                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Unet                   | **[Paper Link](https://arxiv.org/pdf/1505.04597)**                                                                      |
| Unet++                 | **[Paper Link](https://arxiv.org/pdf/1807.10165)**                                                                      |
| FPN                    | **[Link](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)**                                                  |
| DeepLabV3              | **[Paper Link](https://arxiv.org/pdf/1706.05587)**                                                                      |
| DeepLabV3+             | **[Paper Link](https://arxiv.org/pdf/1802.02611)**                                                                      |
| Linknet                | **[Paper Link](https://arxiv.org/pdf/1707.03718)**                                                                      |
| MAnet                  | **[Paper Link](https://ieeexplore.ieee.org/abstract/document/9201310)**                                                 |
| PAN                    | **[Paper Link](https://arxiv.org/pdf/1805.10180)**                                                                      |
| Segformer (NeurIPS 21) | **[Paper Link](https://proceedings.neurips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)** |

<!-- Version 2 -->
| Baseline Model         | Paper                                                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Unet                   | Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Cham: Springer international publishing.                                                                      |
| Unet++                 | Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018, September). Unet++: A nested u-net architecture for medical image segmentation. In International workshop on deep learning in medical image analysis (pp. 3-11). Cham: Springer International Publishing.                                                                      |
| FPN                    | Kirillov, A., He, K., Girshick, R., & Dollár, P. (2017, July). A unified architecture for instance and semantic segmentation. In Computer Vision and Pattern Recognition Conference. CVPR.                                                 |
| DeepLabV3              | Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587.                                                                      |
| DeepLabV3+             | Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) (pp. 801-818).                                                                      |
| Linknet                | Chaurasia, A., & Culurciello, E. (2017, December). Linknet: Exploiting encoder representations for efficient semantic segmentation. In 2017 IEEE visual communications and image processing (VCIP) (pp. 1-4). IEEE.                                                                      |
| MAnet                  | Fan, T., Wang, G., Li, Y., & Wang, H. (2020). Ma-net: A multi-scale attention network for liver and tumor segmentation. Ieee Access, 8, 179656-179665.                                                 |
| PAN                    | Li, H., Xiong, P., An, J., & Wang, L. (2018). Pyramid attention network for semantic segmentation. arXiv preprint arXiv:1805.10180.                                                                      |
| Segformer (NeurIPS 21) | Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34, 12077-12090. |

<!-- 
Execution environment (Python and key library versions, requirements, etc.)
How to run (training/evaluation commands, expected output)
Reproducibility scope (see Section 2)
AI tools used (see Section 5)
Baseline sources (see Section 6)

2. Reproducibility Scope
The TA must be able to run your project end-to-end after downloading it.

Default: Include the training dataset and training pipeline so that everything from training to evaluation can be reproduced.
Exception: If full training reproduction is impractical due to environment, time, or resource constraints, submitting the trained model weights + evaluation dataset + inference/evaluation code is acceptable, as long as your results can be reproduced.
In this case, please state in the README why training is not included and exactly which results can be reproduced from your submission.

3. Large Files (Weights/Datasets)
Large weights or datasets do not need to be included directly in the zip; a download script or link is acceptable.
If using a link, make sure access permissions are correct (publicly viewable/downloadable). If the TA cannot access it, it will be treated as a failed reproduction.

4. Presentation Slides
For fairness, presentation slides must also be submitted by June 9, together with the code.

5. Use of AI / Coding Agents
The use of coding agents and AI tools for implementation support is permitted.
However, please specify in the README which tools you used and how/where you applied them.

6. Baseline / Existing Code
If you use a baseline model or existing code for comparison experiments, you do not need to submit the full code — simply cite the source (paper, repo link, etc.)
-->
