Computer Vision

The whole experiment is performed on NVIDIA 3090 gpu.

### To download opensurface-data
```
mkdir opensurfaces-data
cd opensurfaces-data

wget http://labelmaterial.s3.amazonaws.com/release/opensurfaces-release-0.zip
wget http://labelmaterial.s3.amazonaws.com/release/process_opensurfaces_release_0.py

unzip opensurfaces-release-0.zip

python3 process_opensurfaces_release_0.py
```sh

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