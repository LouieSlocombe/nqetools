# NQE tools

A package to help run NQE (nuclear quantum effects) calculations based on [i-pi](https://github.com/i-pi/i-pi).

# Install instructions

Everything except PLUMED comes from one conda environment file. Work from a
fresh environment — mixing this stack into an existing one causes hard-to-debug
conflicts between the CUDA builds of PyTorch and OpenMM.

```
conda env create -f build_tools/environment.yml
conda activate nqetools
pip install --no-deps -e .
```

The `--no-deps` is deliberate. Every runtime dependency is already installed by
conda-forge, and several of them (`ambertools`, `nnpops`, `openmm-torch`) have
no PyPI distribution, so letting pip resolve them will fail.

To install without a checkout, into an environment that already has the
dependencies:

```
unset SSH_ASKPASS
pip install git+https://github.com/LouieSlocombe/nqetools.git
```

Python 3.13 or newer is required. `build_tools/environment.yml` pins 3.13
because that is where conda-forge currently has builds for the whole stack —
`ambertools` and `nnpops` in particular lag new releases.

PLUMED is the one piece the environment file cannot fully guarantee. OPES is an
optional PLUMED module and is not necessarily enabled in a packaged build, so if
you are running OPES, check first and build from source if it is missing. See
[PLUMED](#plumed) below.

## GPU builds

`environment.yml` lets conda choose the PyTorch build. To pin a specific CUDA
version, check what you have:

```
nvcc --version
```

then install the matching build before creating the rest of the environment,
for example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

Check the backend can actually see the GPU — this should return `True`:

```
import torch
torch.cuda.is_available()
```

MACE evaluations can be sped up with cuequivariance. It is optional and not a
declared dependency, so install it separately if you want it:

```
pip3 install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12
```

If this fails, it is usually because `libc6` is too old for the prebuilt
wheels. On a machine you administer, upgrading the distribution fixes it; on a
cluster you cannot, so skip cuequivariance there.

## PLUMED

We need to build from source.

### Manual compilation

Needed for OPES. See docs `https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html`.

- MAKE SURE YOUR ENV IS ACTIVATED
- install compilers in conda
  `conda install -c conda-forge gcc_linux-64=13 gxx_linux-64=13 gfortran_linux-64=13 sysroot_linux-64`
- Download from `https://github.com/plumed/plumed2/releases`.
- Extract `tar -xvzf plumed-2.9.3.tgz`
- Change into directory `cd plumed-2.9.3`
- Configure — both flags in one invocation, a second `./configure` discards the
  settings of the first: `./configure --prefix=$HOME/opt --enable-modules=opes`
- `make -j 4`
- `make install`
- The kernel environmental variable must be added to the .bashrc, pointing at
  the install prefix you configured above:

```
export PLUMED_KERNEL=$HOME/opt/lib/libplumedKernel.so
```

### Use conda

`py-plumed` (the Python wrapper, `plumed` on PyPI) is already in
`environment.yml`, and the conda-forge `plumed` build can be added with:

```
conda install plumed -y
```

Whether that build has OPES enabled depends on the package, so check before
relying on it. If it does not, keep the source build above and let the wrapper
find it through `PLUMED_KERNEL`.

You can check which kernel is being picked up:

```
import plumed
plumed.Plumed()
```

You should see something along the lines of

+++ Loading the PLUMED kernel runtime +++

+++ PLUMED_KERNEL="$HOME/opt/lib/libplumedKernel.so" +++

<plumed.Plumed object at 0x7f83768e2a00>

Make sure the path it reports is the one you built, not one inside your conda
env — OPES is an optional module and only the source build has it.

## Reaction paths

Reaction-path work — NEB bands, transition-state and IRC searches, and the
geometry helpers for building flipped end states — lives in
[reactiontools](https://github.com/LouieSlocombe/reactiontools), which
`nqetools` installs as a dependency. Import those functions from there:

```python
import reactiontools as rt

reactant, product = rt.optimise_reactant_product(reactant, product, calc)
neb = rt.prepare_neb(reactant, product, calc, n_images=7)
images = rt.optimise_neb(neb, fmax=0.05)
ts = rt.optimise_ts(rt.get_ts_image(images), calc)
```

These were previously re-exported from `nqetools` itself as `nqe.prepare_neb`,
`nqe.optimise_ts`, `nqe.plot_neb` and friends. They are no longer, so that
there is one copy of the code rather than two drifting apart. `nqe.plot_sella`
is now `rt.plot_irc`; everything else kept its name.

Sella drives the transition-state and IRC searches. It is a `reactiontools`
dependency, so it arrives with that package and does not need installing
separately.

### ORCA calculators and free-energy surfaces

The same applies to the ORCA calculator presets and to reading and plotting
free-energy surfaces, which have now moved across too:

```python
import reactiontools as rt

calc = rt.orca_calc_preset(**rt.orca_preset_dft_gold)

surfaces = rt.sum_hills_files("FES")  # FES0.dat, FES1.dat, ...
rt.plot_fes_1d(surfaces, source_unit="kJ/mol", energy_unit="eV")
```

| Was | Now |
| --- | --- |
| `nqe.orca_calc_preset` | `rt.orca_calc_preset` |
| `nqe.orca_preset_dft_cheap` … `nqe.orca_preset_ccsd_gold` | same names on `rt` |
| `nqe.optimise_atoms` | `rt.orca_optimise_atoms` |
| `nqe.calculate_goat` | `rt.orca_calculate_goat` |
| `nqe.get_fmax` | `rt.get_fmax` |
| `nqe.swap_bonding_configuration` | `rt.swap_bonding_configuration` |
| `nqe.n_plot`, `nqe.ax_plot` | `rt.n_plot`, `rt.ax_plot` |
| `nqe.search_fes_files` | `rt.sum_hills_files` |
| `nqe.load_fes_data` | `rt.as_fes` |
| `nqe.plot_fes_series_1d`, `nqe.plot_fes_series_1d_compare` | `rt.plot_fes_1d` |
| `nqe.plot_fes_contourf`, `nqe.plot_fes_contourf_series`, `nqe.plot_fes_contourf_compare` | `rt.plot_fes_2d` |
| `nqe.plot_fes_contour_compare` | `rt.plot_fes_2d_overlay` |
| `nqe.plot_fes_sep` | `rt.plot_fes_slices` |

Two behaviour changes worth knowing about. `rt.orca_calc_preset` defaults to
`xc='r2SCAN-3c'` with no basis set, where the `nqetools` copy defaulted to
`wB97X`/`def2-SVP` — pass them explicitly to reproduce older numbers. And the
`reactiontools` copy fixes a bug the `nqetools` one had: `f_solv="TOLUENE"` was
silently overwritten with `WATER`, and a named `f_disp` with `D4`.

What stays in `nqetools`: the rate theory (`nqe.wigner_correction`,
`nqe.eckart_correction`, the instanton functions), the Arrhenius, transmission
coefficient and KIE plots, `nqe.get_fes_times`, and everything driving i-PI,
PLUMED input generation and OpenMM.

## Driver installation

We need to install the drivers for the code we want to use.

### I-PI drivers

The drivers have to end up inside the installed `ipi` package, in the same
environment i-pi itself lives in. Activate that environment first, then let the
shell resolve where `ipi` actually is rather than hard-coding a path:

```
conda activate nqetools
IPI_PKG=$(python -c 'import ipi, os; print(os.path.dirname(ipi.__file__))')

git clone https://github.com/i-pi/i-pi.git
make -C i-pi/drivers/f90
cp -r i-pi/drivers "$IPI_PKG"
cp -r i-pi/bin "$IPI_PKG"
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

`conda env create` is usually too slow here, so this builds the stack by hand.
`plumed` on PyPI is the same wrapper conda-forge calls `py-plumed` — install
one, not both.

```
module load mamba/latest
mamba create -n nqetools python=3.13
source activate nqetools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ase mace-torch
tar -xvzf plumed-2.9.3.tgz
cd plumed-2.9.3
./configure --prefix=$HOME/opt --enable-modules=opes
make -j 4
make install
export PLUMED_KERNEL=$HOME/opt/lib/libplumedKernel.so
pip install plumed
pip install i-pi
unset SSH_ASKPASS
pip install git+https://github.com/LouieSlocombe/nqetools.git
```

# ARCHER2 install instructions

> **This path is currently broken.** `cray-python/3.10.10` is below the
> `requires-python = ">=3.13"` floor in `pyproject.toml`, so the final
> `pip install` will refuse outright. Load a Python 3.13+ module if the site
> provides one, or bring your own interpreter (miniforge under `$WORK`). The
> recipe below is otherwise still correct — only the interpreter needs
> replacing.

```
module load PrgEnv-gnu
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load cray-python/3.10.10   # too old — see the note above
python -m venv $WORK/nqetools_env
source $WORK/nqetools_env/bin/activate

export PYTHONUSERBASE=$WORK/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.13/site-packages:$PYTHONPATH
export MPLCONFIGDIR=$WORK/.config/matplotlib

pip install --upgrade pip

tar -xvzf plumed-2.9.3.tgz
cd plumed-2.9.3
./configure --prefix=$WORK/opt --enable-modules=opes
make -j 8
make install
export PLUMED_KERNEL=$WORK/opt/lib/libplumedKernel.so

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
pip install ase mace-torch
pip install plumed
pip install i-pi
unset SSH_ASKPASS
pip install git+https://github.com/LouieSlocombe/nqetools.git
```
