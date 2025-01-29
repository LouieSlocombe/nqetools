# NQE tools
A package to help run NQE (nuclear quantum effects) calculations based on [i-pi](https://github.com/i-pi/i-pi).

# Install instructions
I suggest you work from a fresh environment to prevent issues! 
```
conda create -n ipi_env python=3.12
```
```
conda activate ipi_env
```
Make sure to upgrade conda and pip.
```
conda install anaconda::pip
conda update conda --all
```

Install the basic requirements.
```
conda install conda-forge::numpy conda-forge::scipy conda-forge::matplotlib anaconda::pytest conda-forge::opt_einsum conda-forge::jax conda-forge::jaxlib conda-forge::ml_dtypes anaconda::sympy conda-forge::pyfftw conda-forge::chemiscope -y
```

## ASE
Avoid conda, install conda-forge::sella and conda-forge::ase as it is old.
```
pip install ase sella
```

## MACE
Mace has two options, but the torch option seems best. For model eval and training:
Follow the instructions here. Using conda is probably better `https://pytorch.org/get-started/locally/`. It might look like this:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```
It is worth ensuring the pytorch backend can use the GPU from here. This should return True.
```
import torch
torch.cuda.is_available()
```
Next install mace. Selecting mace-torch:
```
pip install mace-torch
```

## PLUMED
We don't need to build from source. We can use the conda-forge packages. Install a pre-compiled PLUMED binary using the following command
```
conda install -c conda-forge plumed -y
```
Similarly, the python wrappers can be installed with
```
conda install -c conda-forge py-plumed -y
```

You can check if plumed is installed if this returns the plumed path
```
import plumed
plumed.Plumed()
```
Optional? Now the kernel environmental variable should added to the .bashrc.
```
PLUMED_KERNEL=/home/louie/anaconda3/envs/ipi_env/lib/libplumedKernel.so
```

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

## I-PI
Conda is preferred to not mess up the ecosystem.
```
conda install conda-forge::i-pi -y
```



# Resources
https://atomistic-cookbook.org/index.html

https://github.com/i-pi/piqm2023-tutorial

https://github.com/i-pi/tutorials-schools

https://github.com/Sucerquia/ASE-PLUMED_tutorial/tree/master

https://github.com/water-ice-group/plumed_tutorial_mace

# Tools
https://github.com/lab-cosmo/chemiscope

