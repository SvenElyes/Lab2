## Installation

### SPECIFIC CHANGES

To make this repo run on UAM Lab Computers follow following instructions.
-install dependecies (be aware of changes made to the pytorch/torchvision and apex version)
-at the the time of this project, there was a recent change on the apex library, that made the programm not run anymore. Thus a checkout of an later commit was needed. It might be, that this will be resolved/changed in the future. Take care!
-replace certain files in the folders to solve certain import/version problems. You can find the corrected working files ["here"] (https://github.com/SvenElyes/Lab2/tree/master/ResultsLab2/modified_apex_files). The path you have to paste the files can be found in the error message of the specific problem. In our case, it was in the conda environment of the apex library.
-do not forget to download the Base and MEGA models.

## File Replacement
These file replacement are also described in the ["report"](https://github.com/SvenElyes/Lab2/blob/master/report_UAM.pdf)

-["grad_scaler.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/grad_scaler.py) needs to be pasted into apex/transformer/amp

-["utils.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/utils.py) needs to be pasted into apex/transformer

-["mappings.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/grad_scaler.py) needs to be pasted into apex/transformer/amp/tensor\_parallel

-["common.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/common.py) needs to be pasted into apex/transformer/pipeline\_parallel/schedules

-["fwd_bwd_no_pipelining.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/fwd_bwd_no_pipelining.py) needs to be pasted into apex/transformer/pipeline\_parallel/schedules

-["fwd_bwd_pipelining_without_interleaving.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/fwd_bwd_pipelining_without_interleaving.py) needs to be pasted into apex/transformer/pipeline\_parallel/schedules

-["fwd_bwd_pipelining_with_interleaving.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/fwd_bwd_pipelining_with_interleaving.py) needs to be pasted into apex/transformer/pipeline\_parallel/schedules



-["predictor.py"](https://github.com/SvenElyes/Lab2/blob/master/ResultsLab2/modified_apex_files/predictor.py) needs to be pasted itno somewhere else. It has to go to mega.pytorch/demo

## Download Mega and Base Model

["Mega Model"](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view)
["Base Model"](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view)




### Requirements:
- PyTorch 1.2 
- torchvision (0.4.0)from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.2


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name MEGA -y python=3.7
source activate MEGA

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# mega and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python scipy

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
#see if you need the next line!
git checkout aldf804
cd apex
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/Scalsol/mega.pytorch.git
cd mega.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

pip install 'pillow<7.0.0'

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```