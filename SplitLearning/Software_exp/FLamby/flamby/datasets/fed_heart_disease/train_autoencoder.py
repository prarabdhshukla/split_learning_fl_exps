import numpy as np
import torch
import torch.optim as optim
from pca_autoencoder import AutoEncoder
from torch.utils.data import DataLoader as dl
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)
from tqdm import tqdm
import pickle

def main(num_workers_torch, log=False, log_period=10, debug=False, cpu_only=False):
    training_dl = dl(
        FedHeartDisease(train=True, pooled=True, debug=debug),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train=[]
    c=0
    for s, (X,y) in enumerate(training_dl):
        # print(X.shape)
        if c==121:
            continue
        train.append(X.detach().numpy())
        c+=1
    x_train=np.hstack(train)
    x_train=x_train.reshape((4,121,13))
    client_cleveland=x_train[0]
    client_hungarian=x_train[1]
    client_switzerland=x_train[2]
    client_va=x_train[3]
    clients=[client_cleveland,client_hungarian,client_switzerland,client_va]
    ae_objs=[]
    for client in clients:
        ae=AutoEncoder(client.shape[0],client.shape[1])
        ae_objs.append(ae.train(client))
    print('Done training.....')
    client_names=['cleveland','hungarian','switzerland','va']
    for i in range(len(client_names)):
        file_path=r'./autoencoder_obj/{}.pkl'.format(client_names[i])
        with open(file_path,'wb') as client_file:
            pickle.dump(ae_objs[i],client_file,pickle.HIGHEST_PROTOCOL)
    print("Done saving.......")
    

    
    
if __name__=="__main__":
    main(20)
    
        