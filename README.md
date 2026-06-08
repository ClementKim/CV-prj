# Computer Vision Final Project

**This project was completed as the Final Project for `Computer Vision 2026` at GIST, supervised by Professor Kwanyoung Kim.**


### Author

1️⃣ **Junsung Kim** from Electrical Engineering and Computer Science (EECS), Gwangju Institute of Science and Technology (GIST)

2️⃣ **Yongtae Lee** from AI Convergence, Gwangju Institute of Science and Technology (GIST)

3️⃣ **Jaeik Choi** from AI Convergence, Gwangju Institute of Science and Technology (GIST)


### 1. Environment & Requirements

| Item               | Details                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------|
| **OS / Kernel**    | Ubuntu 22.04.3 LTS                                                                                       |
| **GPU**            | NVIDIA A100-SXM4-40GB                                                                                    |
| **Python**         | 3.10.14 (**Conda**)                                                                                      |
| **Core Libraries** | torch (2.10.0) · torchvision (0.25.0) · numpy · timm · pillow · matplotlib · segmentation_models_pytorch |


### 2. How to run

#### 2-1. To download MINC & Opensurface dataset

Download from **[Dataset Download](https://drive.google.com/drive/folders/1BJFKH0cRnT5ZZMy1MzJZQfKx1G0y6-C0?usp=sharing)**.

To extract tar files will be performed automatically by **install.sh**.

#### 2-2. To run
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


### 3. Reproducibility scope

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

Datasets that we used for training and testing are extracted by `install.sh`, and preprocessed by `preprocessing.py`.


### 4. AI tools used
<!-- Edit here -->
For implementation support, we used Claude Code running inside Visual Studio Code. The AI was used to draft code, refactor existing code, debugging, and explain. Specifically, The AI was applied in the following places.

1. It helped convert the original CIFAR-10 classifier, this classifier was implemented for checking if our implementation works or not, into a per-pixel material-segmentation pipeline for the MINC dataset and Opensurface dataset.

2. For the model architecture, it was used to draft and refactor teh dual-encoder --- a pretrained timm ViT-S and torchvision ResNet34 ---, the mixture-of-experts modules, and the segmentation decoder in encoder.py and transformer.py.

3. On the data side, it helped implement collte_fn and the MINC mask handling in preprocessing.py, and it converted process_opensurfaces_release_0.py from Python2 to Python3.

4. We also used AI for debugging and diagnosis: resolving Nan loss, a CUDA/NVML driver-version mismatch, and a non-deterministic nll_loss2d warning, and solving class collapse from severe picel imabalance.

5. Finally, AI helped set up the segmentation_models_pytorch baseline sweep in baseline.py, and write the plotting and visualization utilities plot.py and image.py.


### 5. Baseline sources

We trained and evaluated baseline models via **segmentation_models_pytorch** library.

| Baseline Model | Paper                                                                                                             |
|----------------|-------------------------------------------------------------------------------------------------------------------|
| Unet           | **[Link](https://arxiv.org/pdf/1505.04597)**                                                                      |
| Unet++         | **[Link](https://arxiv.org/pdf/1807.10165)**                                                                      |
| FPN            | **[Link](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)**                                            |
| DeepLabV3      | **[Link](https://arxiv.org/pdf/1706.05587)**                                                                      |
| DeepLabV3+     | **[Link](https://arxiv.org/pdf/1802.02611)**                                                                      |
| Linknet        | **[Link](https://arxiv.org/pdf/1707.03718)**                                                                      |
| PAN            | **[Link](https://arxiv.org/pdf/1805.10180)**                                                                      |
| MAnet          | **[Link](https://ieeexplore.ieee.org/abstract/document/9201310)**                                                 |
| Segformer      | **[Link](https://proceedings.neurips.cc/paper_files/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)** |

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
