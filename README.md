# nqetools
A package to help run NQE calculations based on `https://github.com/i-pi/i-pi`.

# Requirements
Make sure to load your conda environment and upgrade conda `conda update conda --all` and pip `pip install --upgrade pip`. I would install them in this order:
- numpy `conda install numpy`
- matplotlib `conda install matplotlib`
- ase `pip install ase`
- sella `pip install sella`
- ipi `pip install ipi`

# MACE
There are two options to use Mace, but the torch option seems best. For model eval and training:
- Follow the instructions here. Using conda is probably better `https://pytorch.org/get-started/locally/`.
- It might look something like this `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`.
- mace-torch `pip install mace-torch`.

# Install one line
- Conda one line `conda install numpy matplotlib pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
- Pip one line `pip install ase sella ipi mace-torch`
