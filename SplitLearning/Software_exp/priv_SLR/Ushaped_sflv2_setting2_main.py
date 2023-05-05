import os
import random
import string
import socket
import requests
import sys
import threading
import time
import torch
from math import ceil
from torchvision import transforms
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp, split_dataset_cifar_setting2
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import server
import multiprocessing
# from opacus import PrivacyEngine
# from opacus.accountants import RDPAccountant
# from opacus import GradSampleModule
# from opacus.optimizers import DPOptimizer
# from opacus.validators import ModuleValidator
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
from utils import datasets,dataset_settings
import wandb
import pandas as pd
import time 
import torch.nn.functional as F
from utils.split_dataset import DatasetFromSubset



class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label



class Client_try():
    def __init__(self, id, *args, **kwargs):
        super(Client_try, self).__init__(*args, **kwargs)
        self.id = id
        self.front_model = []
        self.back_model = []
        self.losses = []
        self.train_dataset = None
        self.test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.train_iterator = None
        self.test_iterator = None 
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None
        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.train_acc = []
        self.test_acc = []
        self.front_epsilons = []
        self.front_best_alphas = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')


    def backward_back(self):
        self.loss.backward()


    def backward_front(self):
        print(self.remote_activations1.grad)
        self.activations1.backward(self.remote_activations1.grad)


    def calculate_loss(self):
        self.criterion = F.cross_entropy
        self.loss = self.criterion(self.outputs, self.targets)


    def calculate_test_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.test_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.test_acc[-1]}')


    def calculate_train_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.train_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.train_acc[-1]}')

    def create_DataLoader(self, dataset_train,dataset_test,idxs,idxs_test, batch_size, test_batch_size):
        self.train_DataLoader = torch.utils.data.DataLoader(DatasetSplit(dataset_train, idxs), batch_size = batch_size, shuffle = True)
        self.test_DataLoader = torch.utils.data.DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size = test_batch_size, shuffle = True)

    def forward_back(self, out):
        self.back_model.to(self.device)
        self.outputs = self.back_model(out)


    def forward_front(self, type, u_id = None):

        if type == "train":
            
            if self.id == u_id:

                try:
                    self.data, self.targets = next(self.train_iterator)
                except StopIteration:
                    self.train_iterator = iter(self.train_DataLoader)
                    self.data, self.targets = next(self.train_iterator)
            else:
                self.data, self.targets = next(self.train_iterator)

        else:
            self.data, self.targets = next(self.test_iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
        self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        self.remote_activations1 = self.activations1.detach().requires_grad_(True)


    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']

    def idle(self):
        pass


    def load_data(self, dataset, transform):
        try:
            dataset_path = os.path.join(f'data/{dataset}/{self.id}')
        except:
            raise Exception(f'Dataset not found for client {self.id}')
        self.train_dataset = torch.load(f'{dataset_path}/train/{self.id}.pt')
        self.test_dataset = torch.load('data/cifar10_setting2/test/common_test.pt')

        self.train_dataset = DatasetFromSubset(
            self.train_dataset, transform=transform
        )
        self.test_dataset = DatasetFromSubset(
            self.test_dataset, transform=transform
        )


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()


    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()






def generate_random_client_ids_try(num_clients, id_len=4) -> list:
    client_ids = []
    for _ in range(num_clients):
        client_ids.append(''.join(random.sample("abcdefghijklmnopqrstuvwxyz1234567890", id_len)))
    return client_ids 


def generate_random_clients_try(num_clients) -> dict:
    client_ids = generate_random_client_ids_try(num_clients)
    clients = {}
    for id in client_ids:
        clients[id] = Client_try(id)
    return clients



def initialize_client(client, dataset_train,dataset_test,idxs,idxs_test, batch_size, test_batch_size):
    # client.load_data(args.dataset, transform)
    # print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')

    client.create_DataLoader(dataset_train,dataset_test, idxs, idxs_test, batch_size, test_batch_size)


def select_random_clients(clients):
    random_clients = {}
    client_ids = list(clients.keys())
    random_index = random.randint(0,len(client_ids)-1)
    random_client_ids = client_ids[random_index]

    print(random_client_ids)
    print(clients)

    for random_client_id in random_client_ids:
        random_clients[random_client_id] = clients[random_client_id]
    return random_clients


if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)

    mode = "online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning exps", mode = mode)
    wandb.run.name = args.opt_iden

    config = wandb.config          
    config.batch_size = args.batch_size    
    config.test_batch_size = args.test_batch_size        
    config.epochs = args.epochs             
    config.lr = args.lr       
    config.dataset = args.dataset
    config.model = args.model
    config.seed = args.seed
    config.opt = args.opt_iden                              


    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients_try(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')


    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(args.dataset, "data", args.number_of_clients, args.datapoints, args.pretrained)
    dict_users_train , dict_users_test = split_dataset_cifar_setting2(client_ids, train_full_dataset, test_full_dataset)

    # print(f'Random client ids:{str(client_ids)}')
    transform=None


    print('Initializing clients...')
    for i,(_, client) in enumerate(clients.items()):
        initialize_client(client, train_full_dataset, test_full_dataset, dict_users_train[i], dict_users_test[i], args.batch_size, args.test_batch_size)
    # if(args.dataset!='ham10000')
        # class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)
    print('Client Intialization complete.')
    model = importlib.import_module(f'models.{args.model}')

    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
    print('Done')


    for _, client in clients.items():

        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)


    center_model = model.center(pretrained=args.pretrained)
    center_model.to(device)
    center_model_opt = optim.Adam(center_model.parameters(), args.lr)

    common_client = clients[client_ids[1]]
    num_iterations_common = ceil(len(common_client.train_DataLoader.dataset)/args.batch_size)
    num_iterations_unique = ceil(len(clients[client_ids[0]].train_DataLoader.dataset)/args.batch_size)
    print("Unique iterations", num_iterations_unique)
    print(num_iterations_common)
    print("Common iterations", num_iterations_common)
    num_test_iterations = ceil(len(common_client.test_DataLoader.dataset)/args.test_batch_size)
    print("Number of test iterations", num_test_iterations)

    unique_client_id = client_ids[0]
    print(unique_client_id)
    unique_client = clients[unique_client_id]
    unique_client.train_iterator = iter(unique_client.train_DataLoader)


    for epoch in range(args.epochs):

        # print(f"Epoch {epoch}")

        overall_train_acc.append(0)
        for i,(_, client) in enumerate(clients.items()):
            client.train_acc.append(0)
            if i != 0:
                client.train_iterator = iter(client.train_DataLoader)
            
            
        for client_id, client in clients.items():

            # print(client_id)

            #complete local epoch of each client 
            for iteration in range(num_iterations_common):
                # print(iteration)
                if client_id == unique_client_id:
                    client.forward_front("train", u_id = client_id)
                else:
                    client.forward_front("train")
                out_center = center_model(client.remote_activations1)
                out_center_remote = out_center.detach().requires_grad_(True)
                client.forward_back(out_center_remote)
                client.calculate_loss()
                client.backward_back()
                out_center.backward(out_center_remote.grad)
                # client.backward_front()

                client.step_back()

                center_model_opt.step()
                # client.step_front()

                client.zero_grad_back()
                center_model_opt.zero_grad()
                # client.zero_grad_front()
                client.train_acc[-1] += client.calculate_train_acc()
            
            client.train_acc[-1] /= num_iterations_common
            if client_id == unique_client_id:
                print("unique client train accuracy", client.train_acc[-1])
            overall_train_acc[-1] += client.train_acc[-1]

        #merge client back and front layer weights after all clients have completed one local epoch
        #1 epoch completed for client_id client
        overall_train_acc[-1] /= len(clients)
        print(f' Epoch {epoch} Personalized Average Train Acc: {overall_train_acc[-1]}')
        
        params = []
        for _, client in clients.items():
            params.append(copy.deepcopy(client.back_model.state_dict()))
        w_glob_cb = merge_weights(params)

        for _, client in clients.items():
            client.back_model.load_state_dict(w_glob_cb)

        # Testing on every 5th epoch
        if epoch%5 == 0:
            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)
                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.test_iterator = iter(client.test_DataLoader)

                for client_id, client in clients.items():

                    for iteration in range(num_test_iterations):
                        client.forward_front("test")
                        out_center = center_model(client.remote_activations1)
                        out_center_remote = out_center.detach().requires_grad_(True)
                        client.forward_back(out_center_remote)
                        client.test_acc[-1] += client.calculate_test_acc()
       

                    client.test_acc[-1] /= num_test_iterations
                    if client_id == unique_client_id:
                        print("unique client test accuracy", client.test_acc[-1])
                    else:
                        overall_test_acc[-1] += client.test_acc[-1]  #not including test accuracy of unique client

                overall_test_acc[-1] /= (len(clients)-1)
                print(f' Epoch {epoch} Personalized Average Test Acc: {overall_test_acc[-1]}')
            
        
            wandb.log({
                "Epoch": epoch,
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average Test Accuracy": overall_test_acc[-1],  
            })