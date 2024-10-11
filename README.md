# TFT-streamflow
This repository contains the code for running Temporal Fusion Transformers (TFT) for streamflow prediction. The code is part of the experiment conducted for the paper: "Temporal Fusion Transformers for Streamflow Prediction: Value of Combining Attention with Recurrence" by Rasiya Koya and Roy (2024).

## Running the Code

To run the code, use the following steps:

```bash
module purge
module load python/3.10
module load anaconda
module load cuda/11.6

# you might want to run following command
# export NCCL_SOCKET_IFNAME=^docker0,lo 

cd /path/to/working/directory/
conda activate tft_env
srun python TFT_streamflow.py
```

## Required Libraries
All necessary libraries and their versions are listed in the environment.yml file. To create the conda environment, run the following command:

```bash
conda env create -f environment.yml
```
In case any combalitibility issue with cuda arises, the following command might be helpful to install correct cuda versions.

`pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html`

Change `torch=={version}` and `cu{version}`. 

## Citation

If you use this code, please cite the paper below:

```bibtex
@article{rasiyakoya2024temporal,
  title={Temporal Fusion Transformers for streamflow Prediction: Value of combining attention with recurrence},
  author={Rasiya Koya, S. and Roy, T.},
  journal={Journal of Hydrology},
  volume={637},
  pages={131301},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.jhydrol.2024.131301}
}
```


