# NQE tools
A package to help run NQE (nuclear quantum effects) calculations based on [i-pi](https://github.com/i-pi/i-pi).

# Install instructions
You must work from a fresh environment to prevent issues! 
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
conda config --env --add channels pytorch
conda config --env --add channels nvidia
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
conda install conda-forge::pytest conda-forge::numpy conda-forge::scipy conda-forge::matplotlib -y
```
More dependent packages.
```
conda install conda-forge::opt_einsum conda-forge::jax conda-forge::jaxlib conda-forge::ml_dtypes conda-forge::sympy conda-forge::pyfftw conda-forge::chemiscope -y
```
Install jupyterlab
```
conda install conda-forge::jupyterlab -y
```

## ASE and Sella
Conda can be an older version, which is fine.
Start with ase
```
conda install conda-forge::ase -y
```
Then install sella
```
conda config --env --add channels conda-forge
conda install conda-forge::sella -y
```
Alternatively, if it is very slow. But, not suggested:
```
pip install sella
```

## MACE
Mace has two options, but the torch option seems best. For model eval and training:
Follow the instructions here. Using conda is probably better `https://pytorch.org/get-started/locally/`. 
First, check what version of CUDA you have, for example, 12.4.
```
nvcc --version
```
Install pytorch. It might look like this:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```
It is worth ensuring the pytorch backend can use the GPU from here. This should return True.
```
import torch
torch.cuda.is_available()
```
Next, install mace. Selecting mace-torch:
```
pip install mace-torch
```

## PLUMED
We need to build from source.

### Manual compilation
Needed for OPES. See docs `https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html`.

- Download from `https://github.com/plumed/plumed2/releases`.
- Extract `tar -xvzf plumed-2.9.3.tgz`
- Change into directory `cd plumed-2.9.3`
- Configure `./configure --prefix=$HOME/opt`
- Configure `./configure --enable-modules=opes`
- `make -j 4`
- `make install`
- The kernel environmental variable must added to the .bashrc.
```
export PLUMED_KERNEL=$HOME/plumed-2.9.3/src/lib/libplumedKernel.so
```
### Use conda
But we can also use the conda-forge packages.
```
conda install -c conda-forge plumed -y
```
Similarly, the Python wrappers can be installed with
```
conda install -c conda-forge py-plumed -y
```
You can check if plumed is installed if this returns the plumed path
```
import plumed
plumed.Plumed()
```
You should see something along the lines of

+++ Loading the PLUMED kernel runtime +++

+++ PLUMED_KERNEL="/home/louie/plumed-2.9.3/src/lib/libplumedKernel.so" +++

<plumed.Plumed object at 0x7f83768e2a00>


## I-PI
Conda is preferred not to mess up the ecosystem.
```
conda install conda-forge::i-pi -y
```

## NQETOOLS
unset SSH_ASKPASS
```
pip install git+https://github.com/LouieSlocombe/nqetools.git
```


## Driver installation
We need to install the drivers for the codes we want to use.
### I-PI drivers
Make sure to install the drivers in the same env as i-pi and update the path as needed.
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


# Resources
https://atomistic-cookbook.org/index.html

https://github.com/i-pi/piqm2023-tutorial

https://github.com/i-pi/tutorials-schools

https://github.com/Sucerquia/ASE-PLUMED_tutorial/tree/master

https://github.com/water-ice-group/plumed_tutorial_mace

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
