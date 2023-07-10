import os
import csv
import sys
import ast
import pickle
import torch
import numpy as np
from torch import nn
from dataloader import create_dataloader
from models import ReviewClassifier
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

torch.manual_seed(42)


def read_data(path):
    ''' Reads tokenised data from a given file'''
    with open(path, 'r') as file:
        lines = file.readlines()
        data = [ast.literal_eval(line) for line in lines]
    return data

def read_label(path):
    ''' Reads the data label from a given file'''
    labels = np.loadtxt(path, delimiter=',', dtype=int)
    return labels

def train(dataloader, model, criterion, optimizer):
    ''' Training of the NN model'''
    model.train()
    losses, acc = [], []
    for batch in tqdm(dataloader):
        y = batch["Class"]
        logits = model(batch['Review'])
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())
    loss = np.array(losses).mean()
    accuracy = np.array(acc).mean()
    print(f"Train Loss: {loss:.4f} | Train Accuracy: {accuracy:.4f}")
    return loss, accuracy

@torch.no_grad()
def test(dloader, model, criterion):
    ''' Test fn to peform model evaluation'''
    model.eval()
    losses, acc = [], []
    for batch in dloader:
        y = batch["Class"]
        logits = model(batch['Review'])
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        acc.append((preds == y).float().mean().item())
    loss = np.array(losses).mean()
    accuracy = np.array(acc).mean()
    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    return loss, accuracy

def predict(dloader, model):
    ''' Test fn to peform model evaluation'''
    model.eval()
    results = []
    for batch in dloader:
        logits = model(batch['Review'])
        preds = torch.argmax(logits, -1)
        results.append(preds)
    results = torch.cat(results, dim=0)
    return results

def main():
    #Special Tokens
    cls_token = "<CLS>"
    sep_token = "<SEP>"
    pad_token = "<PAD>"
    unk_token = "<UNK>"

    # Loading the word 2 vector model
    w2vmodel = pickle.load(open(r'/Users/mukund/Documents/Study/Spring 2023/MSCI_641/Assignment/Assignment3/data/w2v.model', 'rb'))

    # Adding the special tokens into word2vec vocabulary
    w2vmodel.build_vocab([[cls_token, sep_token, pad_token, unk_token]*20], update = True)

    w2v_vectors = w2vmodel.wv
    embedding_matrix = w2vmodel.wv.vectors
    vocab_size, embedding_dim = embedding_matrix.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = sys.argv[1]

    train_path = os.path.join(data_dir,'train.csv')
    val_path = os.path.join(data_dir,'val.csv')
    test_path = os.path.join(data_dir,'test.csv')
    train_label_path = os.path.join(data_dir,'train_labels.csv')
    val_label_path = os.path.join(data_dir,'val_labels.csv')
    test_label_path = os.path.join(data_dir,'test_labels.csv')

    train_data = read_data(train_path)
    val_data = read_data(val_path)
    test_data = read_data(test_path)
    train_labels = read_label(train_label_path)
    val_labels = read_label(val_label_path)
    test_labels = read_label(test_label_path)

    # 3 activation fns - relu, sigmoid, tanh
    # Hyper parameters - Hidden Units Size, Dropout Rates, Sentence Size
    hyper_parameters = {'dropout_rate': [0.1, 0.3, 0.5], 'hidden_units': [256, 512]}
    param_grid = ParameterGrid(hyper_parameters)
    sentence_length = 25

    results = {}
    results['Activation Fn'] = []
    results['Test Accuracy'] = []
    activation_fns = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
    result_str = ''

    train_dataloader = create_dataloader(train_data, train_labels, w2vmodel, sentence_length, 128, device)
    val_dataloader = create_dataloader(val_data, val_labels, w2vmodel, sentence_length, 128, device)
    test_dataloader = create_dataloader(test_data, test_labels, w2vmodel, sentence_length, 128, device)

    csv_results = []
    for activation_fn in activation_fns:
        max_accuracy = 0
        hp_results = {}
        for parameters in param_grid:
            print('Parameters - {} Activation, {} Dropout rate, {} Hidden Units'.format(activation_fn, parameters['dropout_rate'], parameters['hidden_units']))

            model = ReviewClassifier(parameters['hidden_units'], parameters['dropout_rate'], activation_fns[activation_fn], w2v_vectors, vocab_size, embedding_dim, embedding_matrix, 2, sentence_length)
            model.to(device)
            print(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

            train_results = {}
            train_results['Train Acc'] = []
            train_results['Train Loss'] = []
            train_results['Val Acc'] = []
            train_results['Val Loss'] = []

            for epoch in range(0, 5):
                print('Training...')
                train_loss, train_accuracy = train(train_dataloader, model, criterion, optimizer)
                train_results['Train Acc'].append(train_accuracy)
                train_results['Train Loss'].append(train_loss)
                print("Validating...")
                val_loss, val_accuracy = test(val_dataloader, model, criterion)
                train_results['Val Acc'].append(val_accuracy)
                train_results['Val Loss'].append(val_loss)
                csv_results.append({'Epoch': epoch+1, 'Activation':activation_fn, 'Hidden Units':parameters['hidden_units'], 
                'Dropout': parameters['dropout_rate'], 'Train Accuracy':train_accuracy, 'Val Accuracy': val_accuracy})
            
            print('Training Results', train_results)

            val_accuracy = max(train_results['Val Acc'])
            if val_accuracy >= max_accuracy:
                torch.save(model, 'data/nn_' + activation_fn +'.model')
                max_accuracy = val_accuracy
                best_parameters = parameters
                hp_results['Train Acc'] = train_accuracy
                hp_results['Train Loss'] = train_loss
                hp_results['Val Acc'] = val_accuracy
                hp_results['Val Loss'] = val_loss

            print(train_results)
        result_str += 'Activation Fn = {}, Best Accuracy - {}, Best parameters - Dropout: {} Hidden Units: {}'.format(activation_fn, max_accuracy, best_parameters['dropout_rate'], best_parameters['hidden_units']) + '\n'
        print(result_str)

    test_results =  []
    result = {}
    result['Activation Fn'] = ''
    result['Test Acc'] = 0
    result['Test Loss'] = 0
    models = ['nn_relu', 'nn_tanh', 'nn_sigmoid']
    for model_name in models:
        print(model_name)
        loaded_model = torch.load('data/' + model_name + '.model')
        test_accuracy, test_loss = test(test_dataloader, loaded_model, criterion)
        result['Activation Fn'] = model_name
        result['Test Acc'] = test_accuracy
        result['Test Loss'] = test_loss
        test_results.append(result)

    print(test_results)
    csv_file = 'data/results.csv'
    fieldnames = ['Epoch', 'Activation', 'Hidden Units', 'Dropout', 'Train Accuracy', 'Val Accuracy']

    # Open the CSV file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()
        for data in csv_results:
            # Write the data row
            writer.writerow(data)

if __name__ == '__main__':
    main()
