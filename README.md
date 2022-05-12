# Active Learning for Small Molecule pKa Regression, a Long Way To Go.
Repository for the Active Learning for Small Molecule pKa Regression, a Long Way To Go paper. (insert link to paper when ready)

This repository contains the data utilized for the paper, instructions on how to produce the results, as well as the LaTeX for the paper itself.

We utilized the pKa dataset from  [OPERA](https://github.com/kmansouri/OPERA), and the pKa datasets from the czodrowskilab's [Machine-Learning-Meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa).

## Citation
If you find this work helpful, please cite our paper. The bibtex formatted citation is provided below. (insert bibtex citation)

## Requirements
  - Python 3.8
  - PyTorch 1.10 -- compiled with CUDA
  - pandas 1.3.5
  - RDkit 2022.03.1
  - CUDA 10.1+
  - scikit-learn 1.0.2
  - numpy 1.17.4
  - matplotlib 3.1.3

## Recreating the Figures for the Paper

 1) The results needed to generate the figures are present in results.tar.gz

```
tar -xzf results.tar.gz
```

 2) After extracting, you can follow along with `Making_figures.ipynb`  for the code we used to generate our Figures and Tables in the paper.

## Re-creating our results

 1) First, you will need to extract the data files that we utilized. These are available in data.tar.gz

NOTE -- The data files used for the training of our machine learning models ARE NORMALIZED. This means all of the csv files present in this repository are standardized to have mean 0 and variance 1. We have provided the original data files for the OPERA dataset in `data/opera_pKa_master_unnormalized.csv` and the original data files for the CZO dataset are available in the sdf files at `data/czodrowskilab/*.sdf`

```
tar -xzf data.tar.gz
```

  2) After extracting the data files, you will need the commands to run our scripts. We made the commands we used to train the models needed for each figure available in results_generation_cmds.tar.gz

```
tar -xzf results_generation_cmds.tar.gz
```
  
  3) The extraction generates a series of `generate_figure*.cmds` files. Each of these files contains lines all of the commands needed to generate the data in `results/` that is used in `Making_figures.ipynb` for Figures 1-10 (we will discuss Figure 11 separately).

  WARNING -- any of the `*_greedy_*.cmds` files are assuming that you are able to utilize the `greedy_resubber.py` script, which is formulated to dynamically work with the provided `*.slurm` submission scripts. We HIGHLY recommend running the greedy searches in this manner, since it requires training O(N^2) models for each greedy job, thus without the ability to parallelize these commands it is infeasible to generate this data.
  
  To run the greedy selection jobs modify the `*.slurm` files to your appropriate SLURM partition by specifying your partition after the `#SBATCH -p` line in each file. You then submit the `run_greedy_cpu_al.slurm` script, which will monitor and launch the `run_greedy_selection_al.slurm` script to perform the greedy search.
  
  WARNING -- `run_greedy_cpu_al.slurm` is designed to be run IN SERIAL. As an example `generate_figure4_greedy_data.cmds` contains 3 lines of jobs to be run. They need to be submitted as follows:
  
```
sbatch --array=1 run_greedy_cpu_al.slurm
```
Then, once that finishes
```
sbatch --array=2 run_greedy_cpu_al.slurm
```
and so on, until all of the jobs in the cmds file have been executed.

  4) In order to generate the data for Figure 11 -- first you need to prepare the ordering suggested by a Gaussian Process. This is handled by the `run_gaussian_process_experiment.py` script.

WARNING -- This script will handle both running the 10 fold experiments for the gaussian process, as well as generate the relevant pickle files, AND generate the data files needed in `generate_figure11_*.cmds`. This process is fairly slow and will take on the order of hours to run.

```
python3 run_gaussian_process_experiment.py
```

  5) after completing step 4, you can execute the commands in `generate_figure11_*.cmds`, which will result in you having reproduced all of the available data utilized in this paper.
