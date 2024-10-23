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
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Obtain hidden layer representation from the RNN
        rnn_out, hidden = self.rnn(inputs)  # rnn_out contains all hidden states, hidden contains the last hidden state
        
        # Obtain output layer representations using the last time step's hidden state
        # We take the last output from the sequence (the last time step's hidden state)
        last_output = rnn_out[-1]  # rnn_out[-1] gives us the hidden state from the last time step

        # Apply the fully connected layer to transform the hidden state to the output space
        output = self.W(last_output)

        # Obtain probability distribution using LogSoftmax
        predicted_vector = self.softmax(output)
        
        return predicted_vector


def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]-1)))
    for elt in testing:
        tst.append((elt["text"].split(), int(elt["stars"]-1)))
    return tra, val, tst


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    best_model = None
    last_train_accuracy = 0
    last_validation_accuracy = 0

    training_accuracies = []
    validation_accuracies = []
    training_losses = []
    validation_losses = []

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        
        avg_training_loss = loss_total / loss_count
        training_losses.append(avg_training_loss.item())
        
        print(avg_training_loss)
        print("Training completed for epoch {}".format(epoch + 1))
        
        train_accuracy_epoch = correct / total
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_accuracy_epoch))
        
        training_accuracies.append(train_accuracy_epoch)

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))

        val_loss_total = 0

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            
            example_loss_val = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
            val_loss_total += example_loss_val.data
            
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        
        avg_validation_loss = val_loss_total / len(valid_data)
        
        validation_losses.append(avg_validation_loss.item())
        
        print("Validation completed for epoch {}".format(epoch + 1))
        
        val_accuracy_epoch = correct / total
        
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_accuracy_epoch))
        
        validation_accuracies.append(val_accuracy_epoch)

        # Save the best model
        if val_accuracy_epoch > last_validation_accuracy:
            last_validation_accuracy = val_accuracy_epoch
            best_model = model.state_dict()

        # Patience mechanism
        if val_accuracy_epoch < last_validation_accuracy and train_accuracy_epoch > last_train_accuracy:
            print("Early stopping to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
            break
        else:
            last_validation_accuracy = val_accuracy_epoch
            last_train_accuracy = train_accuracy_epoch

    # Load the best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)

    # Test phase
    model.eval()
    correct = 0
    total = 0
    print("========== Testing started ==========")
    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1
    test_accuracy = correct / total
    print("Test accuracy: {}".format(test_accuracy))

    # write out to results/test.out
    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/test.out', 'w') as f:
        f.write("Test accuracy: {}\n".format(test_accuracy))

    # Plotting the results
    epochs = list(range(1, len(training_accuracies) + 1))

    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy vs epochs
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('rnn_accuracy_curves.png')  # Save the figure as an image
    plt.show()

    # Plot training and validation loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('rnn_loss_curves.png')  # Save the figure as an image
    plt.show()
    #plot learning curve for training loss vs validation accuracy 
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('rnn_learning_curve.png')
    plt.show()
