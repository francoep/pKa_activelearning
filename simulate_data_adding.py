#!/usr/bin/env python3

import os
import sys
import pandas
import torch
import numpy as np
import argparse
import random
import pickle
import torch.nn as nn
import torch.nn.functional as F

#we assume that you are running the model from the main section of this github repository
sys.path.append(os.getcwd())
sys.path.append('models/transformer')

from transformer import make_model
from data_utils import load_data_from_smi_df, construct_loader
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def get_stats(model, loader, lossname):
    '''
    takes the input model and data

    and returns the rmse and pearson's R
    '''
    gold, preds, _ =get_predictions(model,loader, lossname)
    mask = ~np.isnan(gold)
    rmse=np.sqrt(np.mean((preds[mask]-gold[mask])**2))

    if gold[mask].shape[0]>1:
        r=np.corrcoef(preds[mask],gold[mask])[0][1]
    else:
        r=-1
    return rmse,r

def get_predictions(model, loader, lossname):
    '''
    Function takes in a model and a data loader,
    returns an array of the true values, and an array of the predictions, and an array of the sigmas
    '''
    model.eval()
    true=np.array([])
    preds=np.array([])
    sigmas=np.array([])
    for batch in loader:
        adjacency_matrix, node_features, distance_matrix, y = batch
        batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
        y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)

        if lossname == 'evd':
            v=y_pred[:,1]
            alpha=y_pred[:,2]
            beta=y_pred[:,3]
            inverse_evidence = 1. / ((alpha-1)*v)
            s=beta*inverse_evidence
            y_pred=y_pred[:,0]
        elif lossname == 'dist':
            s=y_pred[:,1]
            y_pred=y_pred[:,0]
        else:
            s=np.array(-1)

        true=np.append(true,y.tolist())
        preds=np.append(preds,y_pred.tolist())
        sigmas=np.append(sigmas,s.tolist())
    
    return true, preds, sigmas

def load_initial_model(pt_weightsfile,base_freeze,freeze_til_final,copy_gen,seed,params):
    #setting a new random seed for this particular training.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model=make_model(**params)
    model.cuda()
    model_state_dict = model.state_dict()

    #setting the weights equivalent to the starting points
    pt_state_dict=torch.load(pt_weightsfile)
    for name, param in pt_state_dict.items():
        if 'generator' in name and not copy_gen:
            continue
        elif isinstance(param, torch.nn.Parameter):
            param=param.data
        model_state_dict[name].copy_(param)

    #setting up the xavier normalized parameters for the generator.
    for name, param in model_state_dict.items():
        if 'generator' in name:
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    #making sure that the correct weights will be updraged during training.
    if base_freeze or freeze_til_final:
        for i, child in enumerate(model.children()):
            if i < 2:
                for param in child.parameters():
                    param.requires_grad=False
            elif freeze_til_final:
                proj=list(child.children())[0]
                for j,child2 in enumerate(proj.children()):
                    if j < len(list(proj.children()))-1:
                        for param in child2.parameters():
                            param.requires_grad=False

    #param_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('Trainable Parameters:',param_count)
    
    return model

#defining the loss funtion to use with the 2 number output
def dist_loss(mu,sigma,target):
    clamped_var=torch.clamp(sigma, min=0.00001)
    return torch.mean(torch.log(2*np.pi*clamped_var) / 2 + (mu - target)**2 / (2 * clamped_var))

#defining the loss function to use with the 4 number output
def evidential_loss(mu, v, alpha, beta, targets, lam=0.2, epsilon=1e-4):
    """
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer
    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict
    :return: Loss
    """

    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll

    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

    loss = L_NLL + lam*(L_REG - epsilon)
    return torch.mean(loss)

def train_model(model, seed, epochs, training_dataloader, lossname):
    #setting a new random seed for this particular training.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if lossname=='evd':
        criterion=evidential_loss
    elif lossname=='dist':
        criterion=dist_loss
    else:
        criterion=torch.nn.MSELoss(reduction='mean')

    optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=0.0001,momentum=0.9,weight_decay=0)
    #optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001,weight_decay=0.0)

    model.train()
    model.cuda()
    for epoch in range(epochs):
        for batch in training_dataloader:
            optimizer.zero_grad()
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            out_mask = ~torch.isnan(y)
            y_pred = model(node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            #print(y_pred, y)
            
            #handling the different types of losses
            if lossname=='evd':
                mu=y_pred[:,0].view(y.shape[0],1)
                v=y_pred[:,1].view(y.shape[0],1)
                v=nn.Softplus()(v)+1e-5
                alpha=y_pred[:,2].view(y.shape[0],1)
                alpha=nn.Softplus()(alpha)+1e-5+1
                beta=y_pred[:,3].view(y.shape[0],1)
                beta=nn.Softplus()(beta)+1e-5

                loss=criterion(mu[out_mask],v[out_mask],alpha[out_mask],beta[out_mask],y[out_mask])
            elif lossname=='dist':
                mu=y_pred[:,0].view(y.shape[0],1)
                sigma=y_pred[:,1].view(y.shape[0],1)
                sigma=F.softplus(sigma)

                loss=criterion(mu[out_mask],sigma[out_mask],y[out_mask])
            else:
                loss=criterion(y_pred[out_mask],y[out_mask])

            #print(loss)
            loss.backward()
            optimizer.step()
    return model

parser=argparse.ArgumentParser(description='Simulate adding additional training data for MAT models.')
#arguments for the data to be used
parser.add_argument('--trainfile',type=str,required=True,help='Specify file containing base training data for model. Format -- SMILES,VALUE')
parser.add_argument('--testfile',type=str,required=True,help='File containing the test set for the model. Format -- SMILES,VALUE')
parser.add_argument('--extrafile',type=str,required=True,help='File containing the data that needs to be added into the training data. Format -- SMILES,VALUE.')
parser.add_argument('--test2',default=None,type=str,help='Csv containing the data to be secondarily tested upon. Format SMILES,VALUE. Defaults to not being set.')
parser.add_argument('--epochs',type=int, default=4,help='Number of training epochs. Defaults to 4.')
parser.add_argument('--max_add',type=int,default=None,help='Maximum Number of molecules to add. Defaults to adding every molecule in extrafile.')
parser.add_argument('--outname',type=str,default='simulation_results.pickle',help='Name of the output file. Output is a pickle of the results dictionary --> {<num extra mols added>:[(RMSE,R) per model]}. Defaults to simulation_results.pickle.')

#arguments for setting up the transformer
parser.add_argument('--initial_weights',type=str,required=True, nargs='+',help='File(s) that contain the starting weights for training. Can input any number of starting points and each will be a model that is tracked.')
parser.add_argument('--seeds',type=int,required=True,nargs='+',help='Seeds corresponding to the starting weights for training. Needs to be the same length as args.initial_weights.')
parser.add_argument('--hdim2',type=int,default=0,help='Size of the extra hidden dimension of model construction. Defaults to not existing')
parser.add_argument('--freeze',action='store_true',help='Flag to freeze the weights for the transformer')
parser.add_argument('--freeze_til_final',action='store_true',help='Flag to freeze the weights up until the final layer of the whole model')
parser.add_argument('--copy_gen',action='store_true',help='Flag to copy over the initial weights of the generator.')

#arguments for the loss -- determines how model variance estimation will be done.
parser.add_argument('--loss',choices=['evd','dist','mse'],help='Loss function to use. Either Evidence, Gaussian distribtion, or MSE.')
parser.add_argument('--rand_select_seed',type=int,default=None,help='When set, the model will randomly select data to add to the training set. This is the seed for that procedure.')

args=parser.parse_args()

assert len(args.seeds)==len(args.initial_weights)
if args.loss == 'mse':
    assert len(args.seeds)>1,"Print Need at least 2 seeds for model ensembling"
else:
    assert len(args.seeds)==1,"Only need 1 seed for distribtion-based regression"

#assure random seed is set correctly
if not args.rand_select_seed is None:
    random.seed(args.rand_select_seed)

#loading the various datasets
batch_size=8
train_data, train_labels=load_data_from_smi_df(args.trainfile,one_hot_formal_charge=True,two_d_only=False)
test_data, test_labels=load_data_from_smi_df(args.testfile,one_hot_formal_charge=True,two_d_only=False)
test_loader=construct_loader(test_data,test_labels,batch_size,shuffle=False)
extra_data, extra_labels=load_data_from_smi_df(args.extrafile,one_hot_formal_charge=True,two_d_only=False)
extra_smiles=[x.split(',')[0] for x in open(args.extrafile).readlines()]
if args.test2:
    test2_data, test2_labels=load_data_from_smi_df(args.test2,one_hot_formal_charge=True,two_d_only=False)
    test2_loader=construct_loader(test2_data,test2_labels,batch_size,shuffle=False)

#defining the model parameters
d_atom = train_data[0][0].shape[1]
if args.loss=='evd':
    nout=4
elif args.loss=='dist':
    nout=2
else:
    nout=1

#defining the basic model parameters -- mostly hardcoded
model_params= {
    'd_atom': d_atom,
    'd_model': 1024,
    'N': 8,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.33,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': 0.0,
    'aggregation_type': 'mean',
    'n_output': nout,
    'hdim2': args.hdim2
}

if (not args.max_add is None) and args.max_add >=0:
    total_runs=args.max_add+1
else:
    total_runs=len(extra_data)+1

#main loop of program -- for each additional training... need to figure out and think on how to structure this.
results={'smiles_added':[]}#dictionary to store results: {Num_add_examples:[(RMSE,R) per model]}

if args.test2:
    results['test2']={}#creating a sub-dictionary for the extra test set. This is also {Num_added_examples:[(RMSE,R) per model]}

for i in range(total_runs):
    print(i)
    results[i]=[]
    if args.test2:
        results['test2'][i]=[]
    predictions=[]
    sigma=None

    #now we construct the data loaders for the training set & withheld set
    train_loader=construct_loader(train_data, train_labels, batch_size)
    extra_loader=construct_loader(extra_data, extra_labels, batch_size, shuffle=False)
    print(len(train_data),len(extra_data))

    for j,(init_w,s) in enumerate(zip(args.initial_weights,args.seeds)):
        base_model=load_initial_model(init_w,args.freeze,args.freeze_til_final, args.copy_gen,s,model_params)
        if i==0 and j==0:
            print(base_model)
        print(f'Training Model: {j}')

        trained_model=train_model(base_model, s, args.epochs, train_loader, args.loss)
        #logging the trained performance on the test set.
        rmse,r = get_stats(trained_model,test_loader,args.loss)
        results[i].append((rmse,r))

        #logging the trained performance on the test2 set if it exists.
        if args.test2:
            rmse2,r2 = get_stats(trained_model,test2_loader,args.loss)
            results['test2'][i].append((rmse2,r2))

        #gathering the predictions on the extra data
        if args.loss=='mse':
            _,extra_pred, _=get_predictions(trained_model,extra_loader,args.loss)
            predictions.append(extra_pred)
        else:
            #the other two losses we care about the output sigma instead of the predictions
            _,_,extra_pred=get_predictions(trained_model,extra_loader,args.loss)
            sigmas=extra_pred

        #ensuring that the models are removed from memory
        base_model=None 
        trained_model=None
        torch.cuda.empty_cache()

    #after we have the predictions for each model, we need to calculate the variances in each prediction and find the maximal value
    if args.loss=='mse':
        predictions=np.stack(predictions).T
        sigmas=np.var(predictions,axis=1)

    print(sigmas.shape)
    #now we select the maximal predicted sigma
    if not(i==total_runs-1):
        if args.rand_select_seed is None:
            index=np.argmax(sigmas)
            print(sigmas[index], index, extra_smiles[index])
            #print(sigmas)
        else:
            index=random.randint(0,sigmas.shape[0]-1)
            print('Random Selection:',sigmas[index],index,extra_smiles[index])
            #print(sigmas)

        results['smiles_added'].append(extra_smiles.pop(index))
        train_data.append(extra_data.pop(index))
        train_labels.append(extra_labels.pop(index))
    print(len(train_data),len(extra_data))

#after the loop is completed, dump the stored results.
with open(args.outname,'wb') as outfile:
    pickle.dump(results,outfile)