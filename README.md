# Computer Vision Final Project

## 1. Environment & Requirements

| Item               | Details                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------|
| **OS / Kernel**    | Ubuntu 24.04.4 LTS                                                                                  |
| **GPU**            | NVIDIA GeForce RTX 3090 (24GB VRAM)                                                                 |
| **Python**         | 3.10.14 (`Conda`)                                                                                   |
| **Core Libraries** | torch · torchvision · numpy · timm · pillow · matplotlib                                            |


## 2. How to run

### 2-1. To download MINC & Opensurface dataset

Download from **[Dataset Download](https://drive.google.com/drive/folders/1BJFKH0cRnT5ZZMy1MzJZQfKx1G0y6-C0?usp=sharing)**. Extracing tar files is going to be performed by **install.sh**.

### 2-2. To run
```sh
source install.sh

# With full training
source run.sh true # Currently run_train.sh

# With check point
source run.sh false # Currently run_train.sh
```

## 3. Reproducibility scope

For the reproducibility, we implemented reproducibility code in **main.py** and provided model checkpoint on Attach link.

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

**Attach tables if it is possible**

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