import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt


unk = '<UNK>'
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.dropout = nn.Dropout(p=0.3)
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        output = self.W1(input_vector)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.W2(output)
        predicted_vector = self.softmax(output)
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))

    return tra, val

def save_res(fn, output):
    with open(fn, "a") as file:
        file.write("Epochs\tTraining_Accuracy\t\tTraining_Time\t\tValidation_Accuracy\t\tValidation_Time\n")  
        for epoch, training_accuracy, training_time, validation_accuracy, validation_time in output:
            file.write(f"{epoch}\t\t{training_accuracy:.4f}\t\t\t{training_time:.4f}\t\t\t{validation_accuracy:.4f}\t\t\t{validation_time:.4f}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()
    
    output = []
    train_accuracies = []
    val_accuracies = []
    epochs = []
    train_losses = []  # To track training loss for each epoch
    val_losses = []    # To track validation loss for each epoch
    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) 
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    print("========== Training for {} epochs ==========".format(args.epochs))

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) 
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        train_acc = correct / total
        train_loss = loss.item()  # Get the training loss for this epoch
        train_losses.append(train_loss)
        train_time = time.time() - start_time
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Training loss for epoch {}: {}".format(epoch + 1, loss.item()))  # Print the loss

        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size

        val_acc = correct / total
        val_loss = loss.item()  # Get the validation loss for this epoch
        val_losses.append(val_loss)
        val_time = time.time() - start_time
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_acc))
        print("Validation time for this epoch: {}".format(val_time))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        epochs.append(epoch + 1)
        output.append((epoch + 1, train_acc, train_time, val_acc, val_time))

    if args.test_data:
        print("========== Evaluating on Test Data ==========")
        with open(args.test_data) as test_f:
            test_data = json.load(test_f)

        tes = []
        for elt in test_data:
            tes.append((elt["text"].split(), int(elt["stars"] - 1)))
        print(f"Number of records in test data: {len(tes)}")
        
        test_data = convert_to_vector_representation(tes, word2index)
        model.eval()

        correct = 0
        total = 0
        minibatch_size = 16
        N = len(test_data)
        
        with torch.no_grad():  
            for minibatch_index in tqdm(range(N // minibatch_size)):
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1

        test_accuracy = correct / total
        print(f"Test accuracy: {test_accuracy:.4f}") 
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('ffnn1_accuracy_curves.png')
    plt.close()

    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('ffnn1_loss_curves.png')
    plt.close()

    # plot learning curve for training loss vs validation accuracy 
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('ffnn1_learning_curve.png')
    plt.close()

    save_res("tests_fnn1.out", output) 





