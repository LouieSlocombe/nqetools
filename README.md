# NQE tools
A package to help run NQE (nuclear quantum effects) calculations based on [i-pi](https://github.com/i-pi/i-pi).

# Install instructions
I suggest you work from a fresh environment to prevent issues! 
```
conda create -n ipi_env
conda activate ipi_env
```
Make sure to upgrade conda and pip.
```
conda update conda --all
pip install --upgrade pip
```

Install the basic requirements.
```
conda install numpy matplotlib
pip install ase sella ipi
```

## MACE
Mace has two options, but the torch option seems best. For model eval and training:
Follow the instructions here. Using conda is probably better `https://pytorch.org/get-started/locally/`. It might look like this:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
Selecting mace-torch:
```
pip install mace-torch
```

## PLUMED
We don't need to build from source. We can use the conda-forge packages.
```
conda install -c conda-forge plumed py-plumed
```
You can check if plumed is installed if this returns the plumed path
```
import plumed
plumed.Plumed()
```
Now the kernel environmental variable should added to the .bashrc.
```
PLUMED_KERNEL=/home/louie/anaconda3/envs/ipi_env/lib/libplumedKernel.so
```

# Resources
https://atomistic-cookbook.org/index.html

https://github.com/i-pi/piqm2023-tutorial

https://github.com/i-pi/tutorials-schools

https://github.com/Sucerquia/ASE-PLUMED_tutorial/tree/master

https://github.com/water-ice-group/plumed_tutorial_mace

# Tools
https://github.com/lab-cosmo/chemiscope

