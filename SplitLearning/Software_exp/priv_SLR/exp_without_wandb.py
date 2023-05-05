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
from utils.split_dataset import split_dataset
from utils.client_simulation import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import server
import multiprocessing
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
import pandas as pd



def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    client.load_data(args.dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


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


def plot_class_distribution(clients, dataset, batch_size, epochs, opt, client_ids):
    class_distribution=dict()
    number_of_clients=len(client_ids)
    if(len(clients)<=20):
        plot_for_clients=client_ids
    else:
        plot_for_clients=random.sample(client_ids, 20)
    
    fig, ax = plt.subplots(nrows=(int(ceil(len(client_ids)/5))), ncols=5, figsize=(15, 10))
    j=0
    i=0
    #plot histogram
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        class_distribution[client_id]=df['labels'].value_counts().sort_index()
        df['labels'].value_counts().sort_index().plot(ax = ax[i,j], kind = 'bar', ylabel = 'frequency', xlabel=client_id)
        j+=1
        if(j==5 or j==10 or j==15):
            i+=1
            j=0
    fig.tight_layout()
    plt.show()
    plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  

    max_len=0
    #plot line graphs
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
        max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    plt.xticks(np.arange(0,10))
    plt.ylim(0, max_len)
    plt.legend()
    plt.show()
    plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution
 

if __name__ == "__main__":    





    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)
    

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []
    overall_loss = []

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')
    
    split_dataset(args.dataset, client_ids)

    print(f'Random client ids:{str(client_ids)}')
    transform=None



    print('Initializing clients...')
    for _, client in clients.items():
        (initialize_client(client, args.dataset, args.batch_size, args.test_batch_size, transform))
    class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt, client_ids)
    print('Client Intialization complete.')
    model = importlib.import_module(f'models.{args.model}')


    for _, client in clients.items():
        client.front_model = model.front()
        client.back_model = model.back()
    print('Done')

    if args.server_side_tuning:
        dummy_client_id = client_ids[0]
        client_ids = client_ids[1:]
        dummy_client = clients[dummy_client_id]
        clients.pop(dummy_client_id)


    if not args.disable_dp:
        print("DP enabled")
        for _, client in clients.items():
            client.front_privacy_engine = PrivacyEngine()

    for _, client in clients.items():

        # client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=0.001, momentum=0.9)
        # client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=0.001, momentum=0.9)
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=0.001)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=0.001)

    if args.server_side_tuning:
        dummy_client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        dummy_client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)


    if not args.disable_dp:
        for _, client in clients.items():
            client.front_model, client.front_optimizer, client.train_DataLoader = \
                client.front_privacy_engine.make_private(
                module=client.front_model,
                data_loader=client.train_DataLoader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                optimizer=client.front_optimizer,
            )


    first_client = clients[client_ids[0]]
    num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)
    num_test_iterations = ceil(len(first_client.test_DataLoader.dataset)/args.batch_size)

    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_model = model.center()
        s_client.center_model.to(device)
        # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=0.001, momentum=0.9)
        s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), lr=0.001)


    for epoch in range(args.epochs):

        overall_train_acc.append(0)
        for _, client in clients.items():
            client.train_acc.append(0)
            client.iterator = iter(client.train_DataLoader)
            client.running_loss = 0
            

        for iteration in range(num_iterations):
            print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')

            for _, client in clients.items():
                client.forward_front()

            for client_id, client in sc_clients.items():
                client.remote_activations1 = clients[client_id].remote_activations1
                client.forward_center()

            for client_id, client in clients.items():
                client.remote_activations2 = sc_clients[client_id].remote_activations2
                client.forward_back()

            for _, client in clients.items():
                client.calculate_loss()

            for _, client in clients.items():
                client.backward_back()

            for client_id, client in sc_clients.items():
                client.remote_activations2.grad = clients[client_id].remote_activations2.grad
                client.backward_center()

            for client_id, client in clients.items():
                client.remote_activations1.grad = sc_clients[client_id].remote_activations1.grad
                client.backward_front()

            for _, client in clients.items():
                client.step()
                client.zero_grad()

            #merge grads uncomment below
            # params = []
            # for _, client in sc_clients.items():
            #     params.append(client.center_model.parameters())
            # merge_grads(params)

            for _, client in sc_clients.items():
                client.center_optimizer.step()
                client.center_optimizer.zero_grad()

            #merge weights below uncomment 
            params = []
            for _, client in sc_clients.items():
                params.append(client.center_model.parameters())

            merge_weights(params)

            for _, client in clients.items():
                client.running_loss += client.loss
                client.train_acc[-1] += client.calculate_train_acc()

        for c_id, client in clients.items():
            client.train_acc[-1] /= num_iterations
            overall_train_acc[-1] += client.train_acc[-1]

        overall_train_acc[-1] /= len(clients)
        print(f' Overall Train Acc: {overall_train_acc[-1]}')

        if not args.disable_dp:
            for _, client in clients.items():
                front_epsilon, front_best_alpha = client.front_privacy_engine.accountant.get_privacy_spent(delta=args.delta)
                client.front_epsilons.append(front_epsilon)
                client.front_best_alphas.append(front_best_alpha)
                print(f"([{client.id}] ε = {front_epsilon:.2f}, δ = {args.delta}) for α = {front_best_alpha}")


        if args.server_side_tuning:
            dummy_client.iterator = iter(dummy_client.train_DataLoader)
            dummy_client.running_loss = 0

            for iteration in range(num_iterations):
                print(f'\r[Server side tuning] Epoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
                # forward prop for front model at dummy client
                dummy_client.forward_front()

                # send activations to the server at dummy client
                dummy_client.send_remote_activations1()

                # get remote activations from server at dummy client
                dummy_client.get_remote_activations2()

                # forward prop for back model at dummy client
                dummy_client.forward_back()

                # calculate loss at dummy client
                dummy_client.calculate_loss()

                # backprop for back model at dummy client
                dummy_client.backward_back()

                # send gradients to server
                dummy_client.send_remote_activations2_grads()

                # get gradients from server
                dummy_client.get_remote_activations1_grads()

                # backprop for front model at dummy client
                dummy_client.backward_front()

                # update weights of both front and back model at dummy client
                dummy_client.step()

                # zero out all gradients at dummy client
                dummy_client.zero_grad()

                # add losses for each iteration
                dummy_client.running_loss += dummy_client.loss
        
        overall_loss.append(0)
        avg_loss = 0

        for _, client in clients.items():
            loss = client.running_loss/num_iterations
            client.losses.append(loss)
            overall_loss[-1] += loss

        overall_loss[-1] /= len(clients) #(Modified)
        


            # train_acc = 0
            # # average out accuracy of all random_clients
            # for _, client in random_clients.items():
            #     train_acc += client.train_acc[-1]
            # train_acc = train_acc/args.number_of_clients
            # overall_acc.append(train_acc)

        # Testing
        with torch.no_grad():
            test_acc = 0
            overall_test_acc.append(0)
            for _, client in clients.items():
                client.test_acc.append(0)
                client.iterator = iter(client.test_DataLoader)
            for iteration in range(num_test_iterations):
 
                for _, client in clients.items():
                    client.forward_front()

                for client_id, client in sc_clients.items():
                    client.remote_activations1 = clients[client_id].remote_activations1
                    client.forward_center()

                for client_id, client in clients.items():
                    client.remote_activations2 = sc_clients[client_id].remote_activations2
                    client.forward_back()

                for _, client in clients.items():
                    client.test_acc[-1] += client.calculate_test_acc()

            for _, client in clients.items():
                client.test_acc[-1] /= num_test_iterations
                overall_test_acc[-1] += client.test_acc[-1]

            overall_test_acc[-1] /= len(clients)
            print(f' Overall Test Acc: {overall_test_acc[-1]}')
        

    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    server side tuning: {args.server_side_tuning},
                    sigma: {args.sigma}, delta: {args.delta}'''



    for client_id, client in clients.items():
        plt.plot(list(range(args.epochs)), client.train_acc, label=f'{client_id} (Max:{max(client.train_acc):.4f})')
    plt.plot(list(range(args.epochs)), overall_train_acc, label=f'Average (Max:{max(overall_train_acc):.4f})')
    plt.title(f'{args.number_of_clients} Clients: Train Accuracy vs. Epochs')
    plt.ylabel('Train Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ioff()
    plt.savefig(f'./results/train_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    plt.show()

    for client_id, client in clients.items():
        plt.plot(list(range(args.epochs)), client.test_acc, label=f'{client_id} (Max:{max(client.test_acc):.4f})')
    plt.plot(list(range(args.epochs)), overall_test_acc, label=f'Average (Max:{max(overall_test_acc):.4f})')
    plt.title(f'{args.number_of_clients} Clients: Test Accuracy vs. Epochs')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ioff()
    plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    plt.show()

    if not args.disable_dp:

        X_ = first_client.front_epsilons
        Y_ = overall_test_acc
        X_Y_Spline = make_interp_spline(X_, Y_)
        X_ = np.linspace(min(X_), max(X_), 100)
        Y_ = X_Y_Spline(X_)
        ci = 0.5*np.std(Y_)/np.sqrt(len(X_))
        plt.fill_between(X_, (Y_-ci), (Y_+ci), color='blue', alpha=0.5)
        print(ci)
        plt.plot(X_, Y_)
        plt.title(f'{args.number_of_clients} Accuracy vs. Epsilon')
        plt.ylabel('Average Test Acc.')
        plt.xlabel('Epsilon')
        plt.legend()
        plt.ioff()
        plt.figtext(0.45, -0.06, plot_config, ha="center", va="center", fontsize=10)
        plt.savefig(f'./results/acc_vs_epsilon/{timestamp}.png', bbox_inches='tight')
        plt.close()


    with torch.no_grad():
        random_clients_overall_acc = {}
        for random_client_id in clients:

            random_client = clients[random_client_id]
            src = sc_clients[random_client_id]
            random_client_overall_acc = 0
            random_client.test_acc = []

            
            for _, client in clients.items():

                random_client.test_DataLoader = client.test_DataLoader
                random_client.iterator = iter(random_client.test_DataLoader)
                num_test_iterations = ceil(len(random_client.test_DataLoader.dataset)/args.batch_size)
                random_client.test_acc.append(0)
                for iteration in range(num_test_iterations):

                    print(f'\rClient: {client.id}, Iteration: {iteration+1}/{num_test_iterations}', end='')

                    random_client.forward_front()

                    src.remote_activations1 = random_client.remote_activations1
                    src.forward_center()

                    random_client.remote_activations2 = src.remote_activations2
                    random_client.forward_back()

                    random_client.test_acc[-1] += random_client.calculate_test_acc()

                random_client.test_acc[-1] /= num_test_iterations
                random_client_overall_acc += random_client.test_acc[-1]

            random_client_overall_acc /= len(clients)
            random_clients_overall_acc[random_client_id] = random_client_overall_acc

    for client_id in random_clients_overall_acc:
        print(f'{client_id}: {random_clients_overall_acc[client_id]}')
