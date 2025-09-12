# NQE tools
A package to help run NQE (nuclear quantum effects) calculations based on [i-pi](https://github.com/i-pi/i-pi).

# Install instructions
You must work from a fresh environment to prevent issues! 3.12 required for cuequivariance.
```
conda create -n ipi_env python=3.12
```
Activate the env
```
conda activate ipi_env
```
Add channels in this order
```
conda config --env --add channels conda-forge
```
Best to make them strict
```
conda config --set channel_priority true
```
To check your updated channel list, run:
```
conda config --show channels
```

Make sure to upgrade the conda env to force the channel priority
```
conda update conda --all -y
```

Install the basic requirements.
```
conda install pytest numpy scipy matplotlib opt_einsum jax jaxlib ml_dtypes sympy pyfftw chemiscope jupyterlab -y
```

## ASE and Sella
Conda can be an older version, which is fine.
```
conda install ase sella -y
```
Alternatively, if it is very slow. But, not suggested:
```
pip3 install sella
```

## MACE
Mace has two options, but the torch option seems best. For model eval and training:
Follow the instructions here. Using conda is probably better `https://pytorch.org/get-started/locally/`. 
First, check what version of CUDA you have, for example, 12.9.
```
nvcc --version
```
Install pytorch. It might look like this:
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```
It is worth ensuring the PyTorch backend can use the GPU from here. This should return True.
```
import torch
torch.cuda.is_available()
```
Next, install mace. Selecting mace-torch:
```
pip3 install mace-torch
```
You can speed up evaluations using cuequivariance.
```
pip3 install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
```
If this fails, you should update libc6.
```
sudo apt update
sudo apt full-upgrade -y
sudo apt install --only-upgrade libc6 -y
sudo apt install update-manager-core -y
sudo do-release-upgrade
```
The above is the nuclear option; it won't work on a cluster to which you don't have sudo access.

## PLUMED
We need to build from source.

### Manual compilation
Needed for OPES. See docs `https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html`.
- MAKE SURE YOUR ENV IS ACTIVATED
- install compilers in conda `conda install -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gfortran_linux-64=13 sysroot_linux-64`
- Download from `https://github.com/plumed/plumed2/releases`.
- Extract `tar -xvzf plumed-2.9.3.tgz`
- Change into directory `cd plumed-2.9.3`
- Configure `./configure --prefix=$HOME/opt`
- Configure `./configure --enable-modules=opes`
- `make -j 4`
- `make install`
- The kernel environmental variable must be added to the .bashrc.
```
export PLUMED_KERNEL=$HOME/plumed-2.9.3/src/lib/libplumedKernel.so
```
### Use conda
But we can also use the conda-forge packages.
```
conda install plumed -y
```
Similarly, the Python wrappers can be installed with
```
conda install py-plumed -y
```
You can check if Plumed is installed by checking if this returns the Plumed path
```
import plumed
plumed.Plumed()
```
You should see something along the lines of

+++ Loading the PLUMED kernel runtime +++

+++ PLUMED_KERNEL="/home/louie/plumed-2.9.3/src/lib/libplumedKernel.so" +++

<plumed.Plumed object at 0x7f83768e2a00>

You need to make sure that the path it seems to be pointing to is not in your python bin of the env. But, instead it is the one you built! You will need this as OPES is an optional module.

## I-PI
Conda is preferred not to mess up the ecosystem.
```
conda install i-pi -y
```

## NQETOOLS
unset SSH_ASKPASS
```
pip install git+https://github.com/LouieSlocombe/nqetools.git
```


## Driver installation
We need to install the drivers for the code we want to use.
### I-PI drivers
Make sure to install the drivers in the same environment as i-pi and update the path as needed.
```
git clone https://github.com/i-pi/i-pi.git
cd /home/louie/i-pi/drivers/f90
make
cp -r /home/louie/i-pi/drivers /home/louie/anaconda3/envs/ipi_env/lib/python3.13/site-packages/ipi
```
Need to add the bin
```
cp -r /home/louie/i-pi/bin /home/louie/anaconda3/envs/ipi_env/lib/python3.13/site-packages/ipi
```

### CP2K
The exe will likely be `cp2k.ssmp`
```
conda install conda-forge::cp2k -y
```

### NWChem
The exe will likely be `nwchem`
```
conda install conda-forge::nwchem -y
```

### DFTB+
For DFTB+, the exe will likely be `dftb+`
```
conda install conda-forge::dftbplus -y
```
Set the DFTB_PREFIX environment variable to specify the location of parameter sets:
```
export DFTB_PREFIX=/path/to/your/parameter/sets
```
To make it permanent, you can add this line to your .bashrc or .bash_profile.
You can get the parameters from https://www.dftb.org/parameters/download.html
https://github.com/dftbparams/3ob/releases
```
tar -xf 3ob-3-1.tar.xz
```
Put this in the bashrc
```
export DFTB_PREFIX=$HOME/3ob-3-1/
```

# Resources
https://atomistic-cookbook.org/index.html

https://github.com/i-pi/piqm2023-tutorial

https://github.com/i-pi/tutorials-schools

https://github.com/Sucerquia/ASE-PLUMED_tutorial/tree/master

https://github.com/water-ice-group/plumed_tutorial_mace

https://fhi-aims-club.gitlab.io/tutorials/molecular-dynamics-with-i-pi/

# Tools
https://github.com/lab-cosmo/chemiscope

https://gle4md.org/index.html?page=matrix

# SOL install instructions
```
module load mamba/latest
mamba create -n ipi_env python=3.12
source activate ipi_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ase mace-torch sella pyfftw
tar -xvzf plumed-2.9.3.tgz
cd plumed-2.9.3
./configure --prefix=$HOME/opt
./configure --enable-modules=opes
make -j 4
make install
export PLUMED_KERNEL=$HOME/plumed-2.9.3/src/lib/libplumedKernel.so
pip install plumed
pip install py-plumed
pip install i-pi
pip install chemiscope
```
# ARCHER2 install instructions
```
module load PrgEnv-gnu
module load rocm 
module load craype-accel-amd-gfx90a 
module load craype-x86-milan 
module load cray-python/3.10.10
python -m venv $WORK/ipi_env
source $WORK/ipi_env/bin/activate

export PYTHONUSERBASE=$WORK/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH
export MPLCONFIGDIR=$WORK/.config/matplotlib


pip install --upgrade pip

tar -xvzf plumed-2.9.3.tgz
cd plumed-2.9.3
./configure --prefix=$WORK/opt
./configure --enable-modules=opes
make -j 8
make install
export PLUMED_KERNEL=$WORK/plumed-2.9.3/src/lib/libplumedKernel.so


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install ase mace-torch sella pyfftw
pip install plumed
pip install i-pi chemiscope
unset SSH_ASKPASS
pip install git+https://github.com/LouieSlocombe/nqetools.git
```
