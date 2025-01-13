# OTCliM: generating climatologies of optical turbulence using machine learning

OTCliM is a novel approach for deriving comprehensive climatologies of atmospheric optical turbulence strength ($C_n^2$)
using gradient boosting machines. OTCliM addresses the challenge of efficiently obtaining reliable site-specific $C_n^2$
climatologies near the surface, crucial for ground-based astronomy and free-space optical communication.
Using gradient boosting machines and global reanalysis data, OTCliM extrapolates one year of measured $C_n^2$ into a
multi-year time series.

All details can be found in our paper:

> M. Pierzyna, S. Basu, and R. Saathof, "OTCliM: Generating a near-Surface Climatology of Optical Turbulence
> Strength ($C_n^2$) Using Gradient Boosting," _Artificial Intelligence for the Earth Systems_, 2024, accepted.

## Installation

Set up conda environment with the following command:

```bash
conda env create -f environment.yml
```

Activate the environment with:

```bash
conda activate otclim
```

## Usage

### Configuration

Setup the `config.py` file