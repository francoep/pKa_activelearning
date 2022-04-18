import time
import subprocess
import argparse
import glob
import pickle
import numpy as np
import os
import shutil

parser=argparse.ArgumentParser(description='Script to monitor output files, and submit new jobs once things finish')
parser.add_argument('--start',type=int,default=0,help='Starting number of iterations. Defaults to 0.')
parser.add_argument('--end',type=int,required=True,help='Ending number of iterations.')
parser.add_argument('--root',type=str,default='data/opera/clustered/greedy/divorced',help='Root for the training/testing/withheld files.')
parser.add_argument('--outdir',type=str,default='greedy/',help='Root for the output files from running the script.')
parser.add_argument('--loss',choices=['mse','evd','dist'],help='Loss function to use. Either Evidence, Gaussian distribtion, or MSE.')
parser.add_argument('--test2',default=None,help='Filename for secondary test set to use with the training script.')
parser.add_argument('--split',default='0', type=str, help='Split for the training/withheld/test files. Defaults to 0.')
parser.add_argument('--fingerprint',default='morgan',choices=['rdkit','morgan','atompair','torsions','maccs'],help='Type of fingerprint to use. Defaults to morgan.')
parser.add_argument('--bitsize',type=int,default=2048,help='Bit size for the fingerprint. Defaults to 2048')
parser.add_argument('--fv_size',type=int,default=0,help='Argument to add an extra layer to a final hidden vector of size fv_size. Defaults to 0 and not being used.')
parser.add_argument('--max_gpu',type=int, default=30, help='Argument to set the maximum number of gpu jobs to submit. Defaults to 30.')
parser.add_argument('--epochs',type=int, default=4, help='Number of epochs to train the model for. Defaults to 4.')
parser.add_argument('--MATtransformer',action='store_true',help='Flag to run a MAT transformer model instead of fingerprints.')
args=parser.parse_args()

if args.fingerprint=='maccs':
    assert args.bitsize==167,"MACCSkeys produce an 167 bit vector."
assert args.epochs>0,"Need to train for at least 1 epoch"
assert args.max_gpu>0,"Need to run on at least 1 gpu."

#gathering the withheld files
withheld=glob.glob(args.root+f'*withheld{args.split}*.csv')
n_withheld=len(withheld)
print('START')
#print(withheld)
#print(n_withheld)

withheld=[]
for nmols in range(n_withheld):
    withheld.append(f'{args.root}_withheld{args.split}_mol{nmols}.csv')

testfile=f'{args.root}_test{args.split}.csv'

#checking if we need to update the withheld set.
if not args.start==0:
    #flag that we have already run some number of permuations -- no need to redo them
    check=glob.glob(f'{args.outdir}na*_{args.loss}_mol*.pkl')

    nas=set([x.split(f'{args.outdir}na')[1].split('_')[0] for x in check])
    assert str(args.start) not in nas,"You have some specified start files in output dir. Remove them"

    mnums=set([x.split('_mol')[1].split('.pkl')[0] for x in check])
    tmp=[]

    for item in withheld:
        item_mnum=item.split('_mol')[1].split('.csv')[0]
        if item_mnum not in mnums:
            tmp.append(item_mnum)

    withheld=tmp.copy()
    tmp=None

#creating/submitting/analyzing the active learning jobs.
for i in range(args.start, args.end+1):
    trainfile=f'{args.root}_{i}added_train{args.split}.csv'
    current_withheld_count=len(withheld)
    print(i, current_withheld_count)
    #print(withheld)

    #after finding the correct trainfile -- create the needed commands file
    with open('greedy_sel_al.cmds','w') as outfile:
        for fname in withheld:
            molnumber=fname.split('_mol')[-1].split('.csv')[0]
            towrite=''
            if args.MATtransformer:
                if args.loss=='mse':
                    towrite=f'python3 simulate_data_adding.py --trainfile {trainfile} --testfile {testfile} --extrafile {fname} --loss {args.loss} --epochs {args.epochs} --seeds 0 1 2 3 4 --initial_weights ../MAT/pretrained_weights.pt ../MAT/pretrained_weights.pt ../MAT/pretrained_weights.pt ../MAT/pretrained_weights.pt ../MAT/pretrained_weights.pt --freeze --outname {args.outdir}na{i}_{args.split}_{args.loss}_mol{molnumber}.pkl'
                else:
                    towrite=f'python3 simulate_data_adding.py --trainfile {trainfile} --testfile {testfile} --extrafile {fname} --loss {args.loss} --epochs {args.epochs} --seeds 0 --initial_weights ../MAT/pretrained_weights.pt --freeze --outname {args.outdir}na{i}_{args.split}_{args.loss}_mol{molnumber}.pkl'
            else:
                if args.loss=='mse':
                    towrite=(f'python3 simulate_data_adding_fp.py --trainfile {trainfile} --testfile {testfile} --extrafile {fname} --loss {args.loss} --fingerprint {args.fingerprint} --bitsize {args.bitsize} --num_hidden 1 --hidden_dim_size 1024 --lr 0.0001 --epochs {args.epochs} --seeds 0 1 2 3 4 --outname {args.outdir}na{i}_{args.split}_{args.loss}_mol{molnumber}.pkl --fv_size {args.fv_size}')
                else:
                    towrite=(f'python3 simulate_data_adding_fp.py --trainfile {trainfile} --testfile {testfile} --extrafile {fname} --loss {args.loss} --fingerprint {args.fingerprint} --bitsize {args.bitsize} --num_hidden 1 --hidden_dim_size 1024 --lr 0.0001 --epochs {args.epochs} --seeds 0 --outname {args.outdir}na{i}_{args.split}_{args.loss}_mol{molnumber}.pkl --fv_size {args.fv_size}')

            if args.test2:
                towrite+=f' --test2 {args.test2}'

            outfile.write(towrite+'\n')
    #submitting
    time.sleep(1)
    print('Submitting jobs')
    subprocess.call(f"sbatch --array=1-{current_withheld_count}%{args.max_gpu} run_greedy_selection_al.slurm",shell=True)

    #checking for completion
    finished=False 
    while finished is False:
        time.sleep(10)
        files=glob.glob(f"{args.outdir}na{i}_{args.split}_{args.loss}_*.pkl")
        #print(len(files))
        if len(files)==current_withheld_count:
            finished=True

    #now that we are completed -- we need to find the best performing model & pop its index out & write the next training file
    print('Checking outputs')
    best_i=0
    best_rmse=999
    best_fname=''
    for index,fname in enumerate(files):
        data=pickle.load(open(fname,'rb'))
        print(data[1])
        if args.loss=='mse':
            rmse=np.mean([x[0] for x in data[1]])
        else:
            rmse,r = data[1][0]
        if rmse< best_rmse:
            best_i=index
            best_rmse=rmse
            best_fname=fname
    #print(files)
    selected_mol=int(files[best_i].split('_mol')[-1].split('.pkl')[0])
    selected_index=0
    for j, fname in enumerate(withheld):
        if f'_mol{selected_mol}.csv' in fname:
            selected_index=j
            break
    #print(selected_mol)
    #print(selected_index)
    trainlines=open(trainfile).readlines()
    selected=withheld.pop(selected_index)
    print(best_i, best_rmse, best_fname, selected_index, selected)
    selectedlines=open(selected).readlines()
    with open(f'{args.root}_{i+1}added_train{args.split}.csv','w') as outfile:
        for line in trainlines:
            outfile.write(line)
        for line in selectedlines:
            outfile.write(line)

    print('Deleting unneeded files')
    for fname in files:
        if fname!=best_fname:
            os.remove(fname)

print('FINISHED')
