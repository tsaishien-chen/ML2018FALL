import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 
import sys


from utils import *
from data import HumanDataset
from tqdm import tqdm 
from config import config
from datetime import datetime
from models.model import*
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
#set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


#test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    #sample_submission_df = pd.read_csv("./sample_submission.csv")
    sample_submission_df = pd.read_csv(sys.argv[1])
    #confirm the model converted to cuda
    filenames,labels ,submissions= [],[],[]
    model.cuda()
    model.eval()
    submit_results = []
    for i,(input,filepath) in enumerate(tqdm(test_loader)):
        #change everything to cuda and get only basename
        filepath = [os.path.basename(x) for x in filepath]
        with torch.no_grad():
            image_var = input.cuda(non_blocking=True)
            y_pred = model(image_var)
            label = y_pred.sigmoid().cpu().data.numpy()
            #print(label > 0.5)
            #print (type(label))
            #input()
            #ans = []
            #for j in range(28):
            #    if j in [15,9,10,26,20,24,17,27,16]:
            #        ans.append(label[:,j] > 0.05)
            #    else:
            #        ans.append(label[:,j] > 0.15)

            labels.append(label > 0.15)
            #print (ans)
            #labels.append(np.array([ans]))
            ans.clear()
            filenames.append(filepath)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('./submit/%s_bestf1_submission.csv'%config.model_name, index=None)
    #sample_submission_df.to_csv('./%s_report_submission.csv'%config.model_name, index=None)
# main function
def main():
    fold = 5
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    
    model = get_net()
    model.cuda()

    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().cuda()
    #criterion = FocalLoss().cuda()
    #criterion = F1Loss().cuda()
    start_epoch = 0
    best_loss = 999
    best_f1 = 0
    best_results = [np.inf,0]
    val_metrics = [np.inf,0]

    #print(all_files)
    #test_files = pd.read_csv("./sample_submission.csv")
    test_files = pd.read_csv(sys.argv[1])

    # load dataset
    test_gen = HumanDataset(test_files,config.test_data,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=4)
  
    best_model = torch.load("%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
    #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
    model.load_state_dict(best_model["state_dict"])
    test(test_loader,model,fold)

if __name__ == "__main__":
    main()
