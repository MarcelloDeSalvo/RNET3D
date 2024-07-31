# :star2: A Multi-task 3D Convolutional Neural Network for rodent skull-stripping and ROI segmentation
This project was started in collaboration with Politecnico di Milano and the Mario Negri Institute with the goal of creating a multitask convolutional neural network (CNN) in order to automate and simplify ROI segmentation and skull-stripping on TBI rodents.

![](/images/logo.png)

## :link: Table of content
- [:star2: A Multi-task 3D Convolutional Neural Network for rodent skull-stripping and ROI segmentation](#star2-a-multi-task-3d-convolutional-neural-network-for-rodent-skull-stripping-and-roi-segmentation)
  - [:link: Table of content](#link-table-of-content)
  - [:page\_with\_curl: Intro](#page_with_curl-intro)
  - [:mouse: 3D R-Net Architecture](#mouse-3d-r-net-architecture)
  - [:milky\_way: Project environment and prerequisites](#milky_way-project-environment-and-prerequisites)
  - [:wrench: Project setup](#wrench-project-setup)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up a Virtual Environment](#2-set-up-a-virtual-environment)
    - [3. Configure VS Code for Jupyter Notebook](#3-configure-vs-code-for-jupyter-notebook)
    - [4. Open and Run Jupyter Notebook](#4-open-and-run-jupyter-notebook)
    - [5. Verify TensorFlow with CUDA](#5-verify-tensorflow-with-cuda)
  - [:file\_folder: Project structure](#file_folder-project-structure)

## :page_with_curl: Intro
Rodent models of traumatic brain injury (TBI) are crucial for studying the mechanisms underlying brain injury evolution and long-term outcomes. Magnetic resonance imaging (MRI) is increasingly used in preclinical settings to monitor in-vivo structural damage, as it allows direct comparisons to human data. However, automatic segmentation of brain volumes remains challenging due to the scarcity of rodent brain segmentation methods, making the procedure highly time-consuming and operator-dependent

![](/images/U-Net%20-%20Pipeline.png)
By using Convolutional Neural Networks (CNNs), this study aims to develop an efficient tool for enhancing segmentation quality while minimizing analysis time. We explore the effectiveness of 3D CNN architectures across different species, strains, injury sites, and MRI sequences: using a dataset of rodents with TBI from the controlled cortical impact (CCI) model, we trained and evaluated a multi-task CNN incorporating multi-scale inputs, deep supervision, and attention mechanisms to enhance segmentation performance.

## :mouse: 3D R-Net Architecture
Our proposed multi-task 3D U-Net architecture features:

- **Shared Encoder and Decoder**: A unified pathway for efficient learning, with a total of 7.4 million parameters.
- **Skip Connections**: To transfer matching feature maps from the encoder to the decoder.
- **Self-Attention Layers**: Feature maps in the decoder are processed with self-attention layers for enhanced precision.
- **Deep Supervision**: Intermediate results are summed up separately for each task, ensuring precise and fine-grained 3D segmentation.
- **Multi-scale inputs**: Preserves small or subtle image details that are frequently lost during max-pooling by gradually incorporating multi-scale inputs into the encoder layers.

![](/images/U-Net%20-%20Architecture.png)

## :milky_way: Project environment and prerequisites
Before you begin, ensure you have the following installed on your system:
- Python 3.8+
- Git
- Visual Studio Code
- TensorFlow 2.10 (Windows)
- NVIDIA CUDA Toolkit 11.2 (Ensure compatibility with your GPU)


All other dependecies are listed inside `requirements.txt` file

<p align="left">
  <a href="https://git-scm.com/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/></a>
  <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/></a>
  <a href="https://www.python.org" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/></a>
  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/></a>
  <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/></a>
  <a href="https://www.tensorflow.org" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/></a>
  <a href="https://nipy.org/nibabel/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://nipy.org/nibabel/_static/nibabel-logo.svg" alt="nibabel" width="40" height="40"/></a>
  <a href="https://numpy.org/" target="_blank" rel="noreferrer" style="display:inline-block;"><img src="https://numpy.org/images/logo.svg" alt="numpy" width="40" height="40"/></a>
</p>

## :wrench: Project setup
### 1. Clone the Repository
First, clone the repository from GitHub to your local machine.
```python
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment using VS Code commands.

1. Open VS Code and navigate to the cloned repository folder.

2. Open the Command Palette (`Ctrl+Shift+P`) and run the following commands:
    - `Python: Create Environment`
    - Choose `Venv`
    - Select `requirements.txt` to automatically install dependencies

### 3. Configure VS Code for Jupyter Notebook
Ensure you have the necessary extensions in VS Code:
- Python
- Jupyter

Open VS Code, navigate to the Extensions view by clicking the square icon in the sidebar or pressing `Ctrl+Shift+X`, and search for and install the Python and Jupyter extensions.

### 4. Open and Run Jupyter Notebook
To open and run your Jupyter Notebook:

1. Open the Command Palette (`Ctrl+Shift+P`) and type `Python: Select Interpreter`. Choose the interpreter from your virtual environment (it should be named `venv`).

2. Navigate to the Jupyter Notebook file (e.g., `notebook.ipynb`) within VS Code and open it.

3. Run the notebook cells to execute your TensorFlow code with CUDA support.

### 5. Verify TensorFlow with CUDA
To ensure TensorFlow is using CUDA, run the following code in a Jupyter Notebook cell:
```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
## :file_folder: Project structure
<pre>
Example/        # Contains two labelled mice in nifti format
├── FLASH
└── RARE

Models/         # Contains pre-trained models for various experimental setups
├── rnet_mice.h5/      # Trained on TBI mice (lesion + 3 ventricles)
├── rnet_rats.h5/      # Tained on TBI rats (lesion + 3 ventricles)
└── rnet_da.h5/        # Trained on TBI mice with domain adaptation (lesion + 3 ventricles + cortex + corpus callosum)
 
Src/            # The main folder of the project containing all the source code
├── main_script.py
├── utils.py
└── other_script.py
</pre>

