'''
This script contains all hard-coded paths, as it is designed to run the Gaussian Process to generate both the data orderings, as well as process the data for Experiment 11

WARNING -- This code will take a fair amount of time to execute O(hours)
'''

from collections import defaultdict as ddict, OrderedDict as odict
from typing import Any, Dict, List
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem as Chem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from random import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF
import pickle
import os
import shutil

########
#Setting up the Data handling classes
########

#helper functions
def generate_splits(dataset, training_set_size=10):
    indices = np.arange(dataset.shape[0])
    shuffle(indices)
    return dataset[indices[:training_set_size], :], dataset[indices[training_set_size:], :]

def split_variables(data_array):
    ids, data_array = np.hsplit(data_array, np.array([1]))
    X, y = np.hsplit(data_array, np.array([-1]))
    return ids, X, y

def chunk(data):
    return np.split(data, data.shape[0], axis=0)

def assort_samples(ids, X, y):
    ids = chunk(ids)
    features = chunk(X)
    labels = chunk(y)
    return list(zip(ids, features, labels))


# DATASET CLASS
class Dataset(object):
    def __init__(self, observed_set, unobserved_set):
        ids0, X0, y0 = split_variables(observed_set)
        idst, Xt, yt = split_variables(unobserved_set)
        self.observed = assort_samples(ids0, X0, y0)
        self.unobserved = assort_samples(idst, Xt, yt)
        self.scaler = StandardScaler()
    
    def __len__(self):
        return len(self.observed)

    def get_training_set(self):
        _, X, y = list(zip(*self.observed))
        X = np.concatenate(X, axis=0)
        return self.scaler.fit_transform(X), np.concatenate(y, axis=0)
    
    def get_test_set(self):
        _, X, y = list(zip(*self.unobserved))
        X = np.concatenate(X, axis=0)
        return self.scaler.fit_transform(X), np.concatenate(y, axis=0)
    
    def update_design(self, index):
        id_, x, y = self.unobserved.pop(index)
        self.observed.append((id_, x, y))
    
    def scale_datapoint(self, x_t):
        s = StandardScaler()
        X, _ = list(zip(*self.observed))
        X = np.concatenate(X, axis=0)
        s.fit(X)
        return s.transform(x_t)
#########################################################

###########
# Classes for the Model and Simulation handlers
##########
class Recorder(object):
    def __init__(self):
        self.tape = []
    
    def __call__(self, data):
        self.tape.append(data)

def fix_nans(array):
    nans = np.isnan(array)
    if np.any(nans):
        array[np.where(nans)] = 1e-8
        return array
    else:
        return array

#defining the Gaussian Process
def gp_predictive(y_train, X_train, X_test, kernel_fn, noise_std):
    n = X_train.shape[0]
    K_aa = kernel_fn(X_train, X_train) + ((noise_std ** 2) * np.eye(n))
    K_aa_inv = np.linalg.inv(K_aa)
    K_ba = kernel_fn(X_test, X_train)
    preds = K_ba @ K_aa_inv @ y_train
    return preds, K_aa_inv, K_ba

def run_simulation(
        kernel,
        datasets,
        sigma_noise=1.0,
        max_train_set_size=104,
        evaluation_protocol=None,
        query_selection_callback=None
        ):
    performance_monitor = Recorder()
    for idx, dataset in enumerate(datasets):
        accuracy_ex = []
        accuracy_test = []
        # BEGIN SIMULATION
        print(f"Begining Simulation {idx + 1}...")
        while len(dataset) < max_train_set_size:
            if (len(dataset) % 10) == 0.0:
                print(f"Current design matrix has {len(dataset)} samples...")
            # EVALUATION STEP
            X_train, y_train = dataset.get_training_set()
            X_train = fix_nans(X_train)
            X_test, y_test = dataset.get_test_set()
            X_test = fix_nans(X_test)
            preds, K_inv, K_x = gp_predictive(
                                              y_train,
                                              X_train,
                                              X_test,
                                              kernel,
                                              sigma_noise
                                              )
            mse = mean_squared_error(y_test, preds)
            accuracy_test.append(mse)
            # EVALUATION PROTOCOL
            # > Run if provided
            if evaluation_protocol:
                metrics = evaluation_protocol(
                                              y_train,
                                              X_train,
                                              K_inv,
                                              kernel,
                                              sigma_noise
                                              )
                accuracy_ex.append(metrics)
            # QUERY SELECTION
            # > If none provided, then random selection is performed
            if not query_selection_callback:
                dataset.update_design(0)
            else:
                K_xx = kernel(X_test, X_test)
                index = query_selection_callback(y_train, K_inv, K_x, K_xx)
                dataset.update_design(index)
        # RECORD ROUND PERFORMANCE
        if len(accuracy_ex) == 0:
            performance_monitor(accuracy_test)
        else:
            performance_monitor( (accuracy_test, accuracy_ex) )
    return performance_monitor, datasets

#Information Gain -- selecting the unobserved data point with the highest predicted variance
def InformationGain(y_train, K_inv, K_x, K_xx):
    return np.argmax(np.diag(K_xx - (K_x @ K_inv @ K_x.T)))

class Evaluator(object):
    def __init__(self, dataset_queue):
        self.queue = [dataset for dataset in dataset_queue]
    
    def __call__(self, y_train, X_train, K_inv, kernel, noise_std):
        scaler = StandardScaler()
        scaler.fit(X_train)
        metrics = []
        for X_eval, y_eval in self.queue:
            X_eval = scaler.transform(X_eval)
            X_eval = fix_nans(X_eval)
            preds, *_ = gp_predictive(y_train, X_train, X_eval, kernel, noise_std)
            metrics.append(mean_squared_error(y_eval, preds))
        return tuple(metrics)
#########################################################

#############
#Processing the training data
#############
training_set_path = "data/czodrowskilab/combined_training_datasets_unique.sdf"
novartis_set_path = "data/czodrowskilab/novartis_cleaned_mono_unique_notraindata.sdf"
avlilumove_set_path = "data/czodrowskilab/AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"

train_df = PandasTools.LoadSDF(training_set_path).astype(
        dict(pKa=float, marvin_atom=int, marvin_pKa=float),
        copy=False
        ).set_index('ID', verify_integrity=True)

novartis_df = PandasTools.LoadSDF(novartis_set_path).astype(
        dict(pKa=float, marvin_atom=int, marvin_pKa=float),
        copy=False
        ).set_index('ID', verify_integrity=True)

avlilumove_df = PandasTools.LoadSDF(avlilumove_set_path).astype(
        dict(pKa=float, marvin_atom=int, marvin_pKa=float),
        copy=False
        ).set_index('ID', verify_integrity=True)

#computing the RDkit descriptors
# In many cases NaN
not_used_desc = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']

# Create a descriptor calculator for all RDKit descriptors except the ones above
desc_calc = MolecularDescriptorCalculator([x for x in [x[0] for x in Descriptors.descList] if x not in not_used_desc])

def calc_x_data(mols):
    """Calculates descriptors and fingerprints for an iterable of RDKit molecules"""
    descs = []    # 196/200 RDKit descriptors
    for mol in mols:
        descs.append(desc_calc.CalcDescriptors(mol))
    descs = np.array(descs)
    return descs

#generating the queue of datasets
random.seed(42)
np.random.seed(42)


eval_queue = []
simulator_splits = []
for idx, df in enumerate([train_df, novartis_df, avlilumove_df]):
    ids = df.index.values
    features = calc_x_data(df.ROMol)
    targets = df.pKa.values
    if idx == 0:
        merged = np.hstack([np.expand_dims(ids, axis=0).T, features, np.expand_dims(targets, axis=0).T])
        for _ in range(10):
            obs, unobs = generate_splits(dataset=merged, training_set_size=10)
            simulator_splits.append((obs, unobs))
    else:
        eval_queue.append( (ids, features, targets) )
eval_protocol = Evaluator(eval_queue)
###########################################################

##############
#Running the Simulations
##    WARNING -- This section of code will take a fairly long amount of time to execute!!!
##############
random.seed(2121992)
np.random.seed(2121992)

ig_sim, ig_ds = run_simulation(
                               kernel=RBF(length_scale=np.sqrt(204)),
                               datasets=[Dataset(*split) for split in simulator_splits],
                               sigma_noise=1.0,
                               max_train_set_size=60,
                               evaluation_protocol=None,
                               query_selection_callback=InformationGain
                               )

rand_sim, rand_ds = run_simulation(
                                   kernel=RBF(length_scale=np.sqrt(204)),
                                   datasets=[Dataset(*split) for split in simulator_splits],
                                   sigma_noise=1.0,
                                   max_train_set_size=60,
                                   evaluation_protocol=None,
                                   query_selection_callback=None
                                   )

#saving out the datasets
datasets = {
        "ig": ig_ds,
        "random": rand_ds
}
pickle.dump(datasets, open("results/rbf_simulation_datasets.pkl", "wb"))

#saving out the Molecule IDs that were selected to add to the training data
training_ids = {}
for k in datasets.keys():
    training_ids.update({k: []})
    for i in range(10):
        ids, *_ = list(zip(*datasets[k][i].observed))
        ids = np.concatenate([id[0] for id in ids], axis=0)
        training_ids[k].append(ids.tolist())
pickle.dump(training_ids, open("results/molecule_ids.pkl", "wb"))

#saving out the simulation results
results = {
        "mean": {
                "ig": np.array(ig_sim.tape).mean(axis=0).tolist(),
                "random": np.array(rand_sim.tape).mean(axis=0).tolist()
        },
        "std": {
                "ig": np.array(ig_sim.tape).std(axis=0).tolist(),
                "random": np.array(rand_sim.tape).std(axis=0).tolist()
        }
}
pickle.dump(results, open("results/rbf_run.pkl", "wb"))

##Running the test set evaluations
class Evaluator(object):
    def __init__(self, dataset_queue):
        self.queue = [dataset for dataset in dataset_queue]
    
    def __call__(self, y_train, X_train, kernel, noise_std):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        metrics = []
        for _, X_eval, y_eval in self.queue:
            X_eval = scaler.transform(X_eval)
            X_eval = fix_nans(X_eval)
            preds, *_ = gp_predictive(y_train, X_train, X_eval, kernel, noise_std)
            metrics.append(mean_squared_error(y_eval, preds))
        return tuple(metrics)

def test_sets(datasets, evaluator_protocol, slice_=None):
    kernel = RBF(length_scale=np.sqrt(204))
    mse_across_sims = []
    for ds in datasets:
        _, Xtrain, ytrain = list(zip(*ds.observed))
        Xtrain = np.concatenate(Xtrain, axis=0)
        ytrain = np.concatenate(ytrain, axis=0)
        if slice_:
            Xtrain = Xtrain[:slice_, :]
            ytrain = ytrain[:slice_, :]
        metrics = evaluator_protocol(ytrain, Xtrain, kernel, 2.0)
        mse_across_sims.append(metrics)
    return mse_across_sims


#actually gathering the data
rounds = list(range(10, 60, 1))
rounds.reverse()

test_eval = {
        "novartis": {
                "ig": [],
                "random": []
        },
        "avlilumove": {
                "ig": [],
                "random": []
        }
}
for round in rounds:
    mses_ig = test_sets(
                        ig_ds,
                        evaluator_protocol=Evaluator(dataset_queue=eval_queue),
                        slice_=round
                        )
    mses_rand = test_sets(
                          rand_ds,
                          evaluator_protocol=Evaluator(dataset_queue=eval_queue),
                          slice_=round
                          )
    # Calc stats
    novartis_avg_ig, avlilumove_avg_ig = np.array(mses_ig).mean(axis=0).tolist()
    novartis_std_ig, avlilumove_std_ig = np.array(mses_ig).std(axis=0).tolist()
    novartis_avg_rand, avlilumove_avg_rand = np.array(mses_rand).mean(axis=0).tolist()
    novartis_std_rand, avlilumove_std_rand = np.array(mses_rand).std(axis=0).tolist()
    # Package data
    test_eval["novartis"]["ig"].append((novartis_avg_ig, novartis_std_ig))
    test_eval["novartis"]["random"].append((novartis_avg_rand, novartis_std_rand))
    test_eval["avlilumove"]["ig"].append((avlilumove_avg_ig, avlilumove_std_ig))
    test_eval["avlilumove"]["random"].append((avlilumove_avg_rand, avlilumove_std_rand))

pickle.dump(test_eval, open("results/test_evals.pkl", "wb"))

#######################
#Generating the datasets for use with the other scripts
#######################
if not os.path.isdir('data/czodrowskilab'):
    os.mkdir('data/czodrowskilab')

##FIRST -- generate the files for training/withheld/test/test2

#we need to normalize the training data
scaler=StandardScaler()
train_df=train_df.reset_index()
pk_norms=scaler.fit_transform(np.array(train_df['pKa']).reshape(-1,1))
train_df['pKa_norm']=pk_norms.flatten().tolist()

#now we can write out the training files
for i, mid_list in enumerate(training_ids['ig']):
    train_ids=set(mid_list[:10])
    with open(f'data/czodrowskilab/train{i}.csv','w') as trainfile:
        with open(f'data/czodrowskilab/withheld_{i}.csv','w') as withfile:
            for mol_id,pk,mol in zip(train_df['ID'],train_df['pKa_norm'],train_df['ROMol']):
                smi=Chem.MolToSmiles(mol)
                if mol_id in train_ids:
                    trainfile.write(f'{smi},{pk}\n')
                else:
                    withfile.write(f'{smi},{pk}\n')
    
#now we need to do the same for the Avi & novartis set
nov_norms=scaler.fit_transform(np.array(novartis_df['pKa']).reshape(-1,1))
novartis_df['pKa_norm']=nov_norms.flatten().tolist()
with open(novartis_set_path.replace('.sdf','.csv'),'w') as outfile:
    for pk, mol in zip(novartis_df['pKa_norm'],novartis_df['ROMol']):
        smi=Chem.MolToSmiles(mol)
        outfile.write(f'{smi},{pk}\n')

avi_norms=scaler.fit_transform(np.array(avlilumove_df['pKa']).reshape(-1,1))
avlilumove_df['pKa_norm']=nov_norms.flatten().tolist()
with open(avlilumove_set_path.replace('.sdf','.csv'),'w') as outfile:
    for pk, mol in zip(avlilumove_df['pKa_norm'],avlilumove_df['ROMol']):
        smi=Chem.MolToSmiles(mol)
        outfile.write(f'{smi},{pk}\n')

##Second -- generate the files for the training/withheld with the specific orderings
for key in training_ids:
    if not os.path.isdir(f'data/czodrowskilanb/{key}'):
        os.mkdir(f'data/czodrowskilanb/{key}')

    for i,mid_list in enumerate(training_ids[key]):
        #copy over the initial training file for the 0added
        train_lines=open(f'data/czodrowskilanb/train{i}.csv').readlines()
        
        #writing out the new training file
        with open(f'data/czodrowskilanb/{key}/0added_train{i}.csv','w') as outfile:
            for line in train_lines:
                outfile.write(line)
                
        for j, mol_id in enumerate(mid_list[10:]):
            row=train_df[train_df['ID']==mol_id]
            smi=Chem.MolToSmiles(row['ROMol'].values[0])
            pk=row['pKa_norm'].values[0]
            train_lines.append(f'{smi},{pk}\n')
            with open(f'data/czodrowskilanb/{key}/{j+1}added_train{i}.csv','w') as outfile:
                for line in train_lines:
                    outfile.write(line)

##Lastly -- generate a dummy withheld set
with open('data/czodrowskilanb/dummy_withheld.csv','w') as outfile:
    outfile.write('CCCCCC,999\n')