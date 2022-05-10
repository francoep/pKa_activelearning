#!/usr/bin/env python3

import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
import argparse
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
import random

torch.backends.cudnn.determininistic=True
torch.backends.cudnn.benchmark=False

#functions for loading the data
def load_data(filename,fingerprint,bitsize):
    '''
    This function takes a csv filename and returns the torch data loader for it.
    
    Assumptions:
        filename -- a csv file of SMILE,VALUE
    
    Will calculate the default rdkit fingerprint of each SMILE as X, takes the VALUE as Y.
    '''
    
    df=pd.read_csv(filename,header=None)
    x=df[0]
    y=list(df[1])

    smiles=[thing for thing in x]
    mols=[Chem.MolFromSmiles(thing) for thing in x]

    if fingerprint=='rdkit':
        fps=[Chem.RDKFingerprint(mol,fpSize=bitsize) for mol in mols]
    elif fingerprint=='morgan':
        fps=[Chem.GetMorganFingerprintAsBitVect(mol,2,nBits=bitsize) for mol in mols]
    elif fingerprint=='atompair':
        fps=[Pairs.GetHashedAtomPairFingerprint(mol,nBits=bitsize) for mol in mols]
    elif fingerprint=='torsions':
        fps=[Torsions.GetHashedTopologicalTorsionFingerprint(mol,nBits=bitsize) for mol in mols]
    elif fingerprint=='maccs':
        fps=[MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    
    bits=[]
    for bit_vect in fps:
        tmp=np.zeros((1,))
        DataStructs.ConvertToNumpyArray(bit_vect,tmp)
        bits.append(tmp)
    
    return bits,y,smiles

def create_loader(filename,fingerprint,bitsize,batch_size=20,shuff=False):
    '''
    This function creates a torch Dataloader for the given file.
    '''
    
    bits, y, _ = load_data(filename,fingerprint,bitsize)
    loader=construct_loader(bits,y,batch_size,shuff)

    return loader

def construct_loader(bits,label,batch_size=20,shuff=False):

    bits=torch.Tensor(bits).to('cuda')
    label=torch.Tensor(label).to('cuda')
    dataset=TensorDataset(bits,label)

    return DataLoader(dataset,batch_size,shuffle=shuff)

#class for a basic 4 layer network#functions for getting predictions & calculating the stats we want
def get_stats(model, loader, dist_flag=False, evd_flag=False):
    '''
    takes the input model and data

    and returns the rmse and pearson's R and mean sigma^2
    '''
    gold, preds, sigmas=get_predictions(model,loader,dist_flag)

    mask = ~np.isnan(gold)
    rmse=np.sqrt(np.mean((preds[mask]-gold[mask])**2))

    if gold[mask].shape[0]>1:
        r=np.corrcoef(preds[mask],gold[mask])[0][1]
    else:
        r=-1
    return rmse,r,np.mean(sigmas)

def get_predictions(model, loader, dist_flag=False, evd_flag=False):
    '''
    Function takes in a model and a data loader,
    returns an array of the true values, and an array of the predictions
    '''
    model.eval()
    true=np.array([])
    preds=np.array([])
    sigmas=np.array([])
    for batch in loader:
        X, y = batch
        y_pred=model(X)
        if dist_flag and evd_flag:
            v=y_pred[:,1]
            alpha=y_pred[:,2]
            beta=y_pred[:,3]
            inverse_evidence = 1. / ((alpha-1)*v)
            s=beta*inverse_evidence
            y_pred = y_pred[:,0]
            
        elif dist_flag:
            s=y_pred[:,1]
            y_pred=y_pred[:,0]

        else:
            s=np.array(-1)

        true=np.append(true,y.tolist())
        preds=np.append(preds,y_pred.tolist())
        sigmas=np.append(sigmas,s.tolist())
    return true, preds, sigmas
class Network(nn.Module):
    def __init__(self, outsize,n_hidden,hidden_size,in_features=2048,min_val=1e-5,fv_size=0):
        super(Network,self).__init__()
        self.in_features=in_features
        self.outsize=outsize
        self.n_hidden=n_hidden
        self.hidden_size=hidden_size
        self.min_val=min_val
        self.fv_size=fv_size

        #creating the feedforward layers
        ffn=[nn.Linear(in_features,self.hidden_size)]
        for _ in range(n_hidden):
            ffn.extend([nn.ReLU(), nn.Linear(hidden_size,hidden_size)])

        if fv_size==0:
            ffn.extend([nn.ReLU(), nn.Linear(hidden_size,outsize)])
        else:
            ffn.extend([nn.ReLU(), nn.Linear(hidden_size,fv_size)])
            ffn.extend([nn.ReLU(), nn.Linear(fv_size,outsize)])
        self.ffn=nn.Sequential(*ffn)
        #self.fc1 = nn.Linear(in_features,1024)
        #self.fc2 = nn.Linear(1024,512)
        #self.fc3 = nn.Linear(512,256)
        #self.fc4 = nn.Linear(256,outsize)
    
    def forward(self,x):
        #x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))
        #x=F.relu(self.fc3(x))
        output=self.ffn(x)
        
        if self.outsize==4:
            
            means, loglambdas, logalphas, logbetas = torch.split(output, output.shape[1]//4, dim=1)
            lambdas=nn.Softplus()(loglambdas)+self.min_val
            alphas = nn.Softplus()(logalphas) + self.min_val + 1  # add 1 for numerical contraints of Gamma function
            betas = nn.Softplus()(logbetas) + self.min_val

            #lambdas=F.softplus(loglambdas)+min_val
            #alphas = F.softplus(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
            #betas = F.softplus(logbetas) + min_val

            output = torch.stack((means,lambdas,alphas,betas),dim=2).view(output.size())
        elif self.outsize==2:
            means, confidences = torch.split(output, output.shape[1]//2,dim=1)
            capped_confidences=F.softplus(confidences)

            output = torch.stack((means,capped_confidences),dim=2).view(output.size())

        return output

def initialize_weights(model: nn.Module, pt_filename=None):
    """
    Initializes the weights of a model in place.
    :param model: An nn.Module.
    """

    if pt_filename:
        pt_state_dict=torch.load(pt_filename)
        model_state_dict=model.state_dict()
        for name, param in pt_state_dict.items():
            if isinstance(param, torch.nn.Parameter):
                param=param.data
            model_state_dict[name].copy_(param)
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
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

#functions for getting predictions & calculating the stats we want
def get_stats(model, loader, dist_flag=False, evd_flag=False):
    '''
    takes the input model and data

    and returns the rmse and pearson's R and mean sigma^2
    '''
    gold, preds, sigmas=get_predictions(model,loader,dist_flag)

    mask = ~np.isnan(gold)
    rmse=np.sqrt(np.mean((preds[mask]-gold[mask])**2))

    if gold[mask].shape[0]>1:
        r=np.corrcoef(preds[mask],gold[mask])[0][1]
    else:
        r=-1
    return rmse,r,np.mean(sigmas)

def get_predictions(model, loader, dist_flag=False, evd_flag=False):
    '''
    Function takes in a model and a data loader,
    returns an array of the true values, and an array of the predictions
    '''
    model.eval()
    true=np.array([])
    preds=np.array([])
    sigmas=np.array([])
    for batch in loader:
        X, y = batch
        y_pred=model(X)
        if dist_flag and evd_flag:
            v=y_pred[:,1]
            alpha=y_pred[:,2]
            beta=y_pred[:,3]
            inverse_evidence = 1. / ((alpha-1)*v)
            s=beta*inverse_evidence
            y_pred = y_pred[:,0]
            
        elif dist_flag:
            s=y_pred[:,1]
            y_pred=y_pred[:,0]

        else:
            s=np.array(-1)

        true=np.append(true,y.tolist())
        preds=np.append(preds,y_pred.tolist())
        sigmas=np.append(sigmas,s.tolist())
    return true, preds, sigmas

#some function to tran a model on some data...
def train_model(model,optimizer,loss_function,epochs,loader,loss_type):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            X,y=batch

            if loss_type=='evd':
                y_hat=model(X)
                mu=y_hat[:,0]
                v=y_hat[:,1]
                alpha=y_hat[:,2]
                beta=y_hat[:,3]

                mask=~torch.isnan(y)
                loss=loss_function(mu[mask],v[mask],alpha[mask],beta[mask],y[mask])
            elif loss_type=='dist':
                y_hat=model(X)
                mu=y_hat[:,0]
                sigma=y_hat[:,1]
                mask=~torch.isnan(y)
                loss=loss_function(mu[mask],sigma[mask],y[mask])
            else:
                y_hat=model(X)
                y=y.view(len(y),1)
                mask=~torch.isnan(y)
                loss=loss_function(y_hat[mask],y[mask])

            loss.backward()
            optimizer.step()
    return model

#function to setup our initial model & set the corresponding seed, also returns the loss function
def load_initial_model(loss_type, random_seed, num_hidden, hidden_size,bitsize, min_val, fv_size, pt_filename):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    #defining the base model & loss function
    if loss_type=='evd':
        base_model=Network(4,num_hidden,hidden_size,in_features=bitsize,min_val=min_val,fv_size=fv_size)
        criterion=evidential_loss
    elif loss_type=='dist':
        base_model=Network(2,num_hidden,hidden_size,in_features=bitsize,min_val=min_val,fv_size=fv_size)
        criterion=dist_loss
    else:
        base_model=Network(1,num_hidden,hidden_size,in_features=bitsize,min_val=min_val,fv_size=fv_size)
        criterion=torch.nn.MSELoss(reduction='mean')

    base_model=initialize_weights(base_model,pt_filename=pt_filename)
    base_model.cuda()
    return base_model, criterion


def get_selection_index(array,n,rand=False):
    '''
    Function to get the indices for selection for active learning.

    Arguments:
        1) array is a numpy array of values whose max n values will be selected.
        2) n is an int representing the number of indices to select
        3) rand is a bool to toggle selecting indices from the array at random.

    Returns:
        a sorted array of indices to select from.
    '''
    if n==1:
        if random:
            index=np.array([random.randint(0,array.shape[0]-1)])
        else:
            index=np.array([np.argmax(array)])
    else:
        if random:
            index=np.sort(random.sample(list(range(array.shape[0])),k=n))
        else:
            tmp=n*-1
            index=np.sort(np.argpartition(array,tmp)[tmp:])

    return index

#setting up the arguments of the script
parser=argparse.ArgumentParser(description='Simulate active learning via various model uncertainty estimations')
#arguments for the data to be used
parser.add_argument('--trainfile',type=str,required=True,help='Csv containing the starting training data.')
parser.add_argument('--testfile',type=str,required=True,help='Csv containing the constant testing data.')
parser.add_argument('--extrafile',type=str,required=True,help='Csv containing the data to be added.')
parser.add_argument('--test2',default=None,type=str,help='Csv containing the data to be secondarily tested upon. Defaults to not being set.')
parser.add_argument('--outname',type=str,required=True,help='Name for the output pickle file. Dictionary of n_added:(rmse,r).')
parser.add_argument('--max_add',type=int,default=None,help='Maximum Number of molecule batches to add. Defaults to adding every molecule in extrafile.')
parser.add_argument('--savemodel', type=str, default=None,help='Filename to save the final trained model.')
parser.add_argument('--savepreds', type=str, default=None,help='Filename to save the predictions. Dictionary of n_added:{pred:[predictions], true:[labels]}). Defaults to not saving.')

#arguments for the training of a model
parser.add_argument('--fingerprint',choices=['rdkit','morgan','atompair','torsions','maccs'],help='Type of fingerprint to use.')
parser.add_argument('--bitsize',type=int,default=2048,help='Bit size for the fingerprint. Defaults to 2048')
parser.add_argument('--lr',type=float,default=0.001,help='Optimizer learning rate. Defaults to 0.001')
parser.add_argument('--epochs',type=int,default=200,help='Number of training epochs. Defaults to 200.')
parser.add_argument('--weight_decay',type=float,default=0.0,help='Amount of weight_decay to add to torch optimizer')
parser.add_argument('--num_hidden',type=int,default=2,help='Number of hidden layers in the network. Defaults to 2.')
parser.add_argument('--hidden_dim_size',type=int,default=300,help='Hidden dimension size. Defaults to 300.')
parser.add_argument('--min_val',type=float,default=1e-5,help='Min value for network. Defaults to 1e-5.')
parser.add_argument('--pt_weights',type=str,default=None,nargs='+',help='Filename containing weights to initialize the model to some pre-trained values.')
parser.add_argument('--fv_size',type=int,default=0,help='Argument to add an extra layer to a final hidden vector of size fv_size. Defaults to 0 and not being used.')

#arguments for the loss -- determines how model variance estimation will be done
parser.add_argument('--loss',choices=['evd','dist','mse'],help='Loss function to use. Either Evidence, Gaussian distribtion, or MSE.')
parser.add_argument('--seeds',type=int,required=True,nargs='+',help='Seeds corresponding to the weight initialization for training.')
parser.add_argument('--rand_select_seed',type=int,default=None,help='When set, the model will randomly select data to add to the training set. This is the seed for that procedure.')

#arguments for the selection of molecules to add in active learning
parser.add_argument('--al_bylabel', action='store_true',help='When set selects molecules for Active Learning by max abosulte difference in predicted label.')
parser.add_argument('--n_add',type=int,default=1,help='Number of molecules to add during active learning cycles. Defaults to 1.')

args=parser.parse_args()

assert args.num_hidden>=0,"Can't have negative hidden dimensions"
assert args.hidden_dim_size>0,"Can't have hidden_dim_dize be 0 or less"
assert args.n_add>0,"Must select at least 1 molecule to add during active learning."
assert args.fv_size>=0,"Final Vector size cannot be negative."

if args.loss == 'mse' and not args.al_bylabel:
    assert len(args.seeds)>1,"Print Need at least 2 seeds for model ensembling"
elif args.loss == 'mse':
    assert len(args.seeds)>=1,"Must have at least 1 seed present for al_bylabel with mse."
else:
    assert len(args.seeds)==1,"Only need 1 seed for distribtion-based regression"

if args.fingerprint=='maccs':
    assert args.bitsize==167,"MACCSkeys requires a 167bit fingerprint."

#loading the various datasets
train_data,train_labels,_=load_data(args.trainfile,args.fingerprint,args.bitsize)
test_loader=create_loader(args.testfile,args.fingerprint,args.bitsize) #unchanging so only need to define the loader
extra_data,extra_labels,extra_smiles=load_data(args.extrafile,args.fingerprint,args.bitsize)

if args.test2:
    test2_loader=create_loader(args.test2,args.fingerprint,args.bitsize) #unchanging so only need to define the loader

#assure random seed is set correctly
if args.rand_select_seed is not None:
    random.seed(args.rand_select_seed)

if args.pt_weights is not None:
    assert len(args.pt_weights) == len(args.seeds),"Need the number of pre-trained weights to correspond to the number of seeds you are training."
    pt_weights=[x for x in args.pt_weights]
else:
    pt_weights=[None for x in args.seeds]

#global needed parameters
multi_regress=args.loss=='evd' or args.loss=='dist'
evidence_model=args.loss=='evd'
if (args.max_add is not None):
    assert args.max_add>=1,"If you are setting a maximum number of batches to add, it has to be at least 1."
    total_runs=args.max_add+1
elif args.n_add >1:
    print(args.n_add)
    print(len(extra_data), len(extra_data)%args.n_add)
    if len(extra_data)%args.n_add:
        total_runs=int(len(extra_data)/args.n_add) + 2
    else:
        total_runs=int(len(extra_data)/args.n_add) + 1
    print(total_runs,len(extra_data))
else:
    total_runs=len(extra_data)+1

results={'smiles_added':[]}#dictionary of {Num_added_examples:[(RMSE,R) per model],smiles_added:[ordered list of added smiles]}
pred_results={}#dictionary of {num_added_examples:[pred,true]} <-- only full if the argument file is set.

if args.test2:
    results['test2']={} #adding a sub-dictionary for the extra test set. The sub-dictionary is also {Num_added_examples:[(RMSE,R) per model]}.
    pred_results['test2']={}

#main loop
if args.al_bylabel:
    last_preds=None

for i in range(total_runs):
    print(i)

    if args.n_add>1 and len(extra_data)<args.n_add:
        n_select=len(extra_data)
    else:
        n_select=args.n_add

    results[i]=[]
    if args.savepreds:
        pred_results[i]={'true':[],'predictions':[]}
    if args.test2:
        results['test2'][i]=[]
        if args.savepreds:
            pred_results['test2'][i]={'true':[],'predictions':[]}
    predictions=[]
    sigmas=None

    train_loader=construct_loader(train_data,train_labels,shuff=True)
    extra_loader=construct_loader(extra_data,extra_labels)

    for j, seed in enumerate(args.seeds):
        base_model,criterion=load_initial_model(args.loss,seed,args.num_hidden,args.hidden_dim_size,args.bitsize,args.min_val, args.fv_size, pt_weights[j])
        optimizer=torch.optim.Adam(base_model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        print(f'Training Model: {j}')

        trained_model=train_model(base_model,optimizer,criterion,args.epochs,train_loader,args.loss)
        #logging the trained performance on the test set
        rmse,r,_ = get_stats(trained_model, test_loader,dist_flag=multi_regress, evd_flag=evidence_model)
        results[i].append((rmse,r))
        #if there is a test2 file, we also need to log those results
        if args.test2:
            rmse2,r2,_ = get_stats(trained_model,test2_loader,dist_flag=multi_regress, evd_flag=evidence_model)
            results['test2'][i].append((rmse2,r2))

        #if we want to save the prediction arrays as well
        if args.savepreds:
            t1_labels, t1_pred, _ = get_predictions(trained_model, test_loader, dist_flag=multi_regress, evd_flag=evidence_model)
            pred_results[i]['true']=t1_labels
            if args.loss=='mse':
                pred_results[i]['predictions'].append(t1_pred)
            else:
                pred_results[i]['predictions']=t1_pred

            if args.test2:
                t2_labels, t2_pred, _ = get_predictions(trained_model, test2_loader,dist_flag=multi_regress, evd_flag=evidence_model)
                pred_results[test2][i]['true']=t2_labels
                if args.loss=='mse':
                    pred_results['test2'][i]['predictions'].append(t2_pred)
                else:
                    pred_results['test2'][i]['predictions']=t2_pred

        #gathering the predictions on the extra data
        if args.loss=='mse':
            _, extra_pred, _=get_predictions(trained_model,extra_loader, dist_flag=multi_regress, evd_flag=evidence_model)
            predictions.append(extra_pred)
        else:
            _, extra_pred, sigmas = get_predictions(trained_model,extra_loader, dist_flag=multi_regress, evd_flag=evidence_model)
        
        if args.savemodel and i==total_runs-1:
            print('Saving Model:',args.savemodel)
            torch.save(trained_model.state_dict(),args.savemodel)

        #ensuring that the models are removed from memory
        base_model=None
        trained_model=None
        del base_model
        del trained_model

    if args.loss=='mse' and len(args.seeds)>1:
        #we need to calculate the ensemble variance for each of the predictions
        predictions=np.stack(predictions).T
        extra_pred=np.mean(predictions,axis=1)
        sigmas=np.var(predictions,axis=1)
        print('multiple seeded sigmas')

        if args.savepreds:
            tmp=pred_results[i]['predictions']
            tmp=np.mean(np.stack(tmp).T,axis=1)
            pred_results[i]['predictions']=tmp
            if args.test2:
                tmp=pred_results['test2'][i]['predictions']
                tmp=np.mean(np.stack(tmp).T,axis=1)
                pred_results['test2'][i]['predictions']=tmp
    elif args.loss=='mse':
        print('sigmas just prediction')
        sigmas=extra_pred

        if args.savepreds:
            pred_results[i]['predictions']=pred_results[i]['predictions'][0]
            if args.test2:
                pred_results['test2'][i]['predictions']=pred_results['test2'][i]['predictions'][0]

    print(sigmas.shape)
    
    #selecting the molecule for active learning
    if not (i==total_runs-1):

        #selecting the max index
        if args.al_bylabel:
            tmp=extra_pred.tolist()
            if i==0 or args.rand_select_seed is not None:
                #we have to do the selection randomly.
                max_index=get_selection_index(extra_pred,n_select,rand=True)
                print('Selected',max_index)
            else:
                #selection by largest absolute difference in predicted labels
                assert len(last_preds)==len(extra_pred)
                sigmas=np.abs(extra_pred-last_preds)
                max_index=get_selection_index(sigmas,n_select)
                print('Selected',max_index)

            for i, ind in enumerate(max_index):
                _ = tmp.pop(ind-i)
            last_preds=np.array(tmp)
        else:
            #now we select the maximal predicted sigma
            if args.rand_select_seed is None:
                max_index=get_selection_index(sigmas,n_select)
                print('Selected',max_index)
            else:
                max_index=get_selection_index(sigmas,n_select,rand=True)
                print('Random Selection:',max_index)

        for i,ind in enumerate(max_index):
            results['smiles_added'].append(extra_smiles.pop(ind-i))
            train_data.append(extra_data.pop(ind-i))
            train_labels.append(extra_labels.pop(ind-i))
    print(len(train_data),len(extra_data))

#after simulation is complete, dump the stored results
with open(args.outname,'wb') as outfile:
    pickle.dump(results,outfile)

if args.savepreds:
    with open(args.savepreds,'wb') as outfile:
        pickle.dump(pred_results,outfile)