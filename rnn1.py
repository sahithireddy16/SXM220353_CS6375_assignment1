import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import string
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h, dropout=0.3):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.dropout = nn.Dropout(dropout)

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        z1, hidden = self.rnn(inputs)
        z1 = self.dropout(z1)
        z2 = self.W(z1[-1])
        predicted_vector = self.softmax(z2)
        return predicted_vector

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)

    tra = [(elt["text"].split(), int(elt["stars"]-1)) for elt in training]
    val = [(elt["text"].split(), int(elt["stars"]-1)) for elt in validation]
    tst = [(elt["text"].split(), int(elt["stars"]-1)) for elt in testing]
    return tra, val, tst

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", required=True, help="path to test data")
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    best_model = None
    last_validation_accuracy = 0
    epochs_no_improve = 0
    cycles = args.epochs

    training_accuracies = []
    validation_accuracies = []
    training_losses = []
    validation_losses = []

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        model.train()
        correct = 0
        total = 0
        minibatch_size = 32
        N = len(train_data)
        loss_total = 0
        loss_count = 0

        print(f"Training started for epoch {epoch + 1}")
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.item()
            loss_count += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_accuracy = correct / total
        avg_training_loss = loss_total / loss_count
        training_accuracies.append(train_accuracy)
        training_losses.append(avg_training_loss)
        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy: {train_accuracy:.4f}, Training loss: {avg_training_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        val_loss_total = 0
        for input_words, gold_label in valid_data:
            input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            example_loss_val = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
            val_loss_total += example_loss_val.item()
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        val_accuracy_epoch = correct / total
        avg_validation_loss = val_loss_total / len(valid_data)
        validation_accuracies.append(val_accuracy_epoch)
        validation_losses.append(avg_validation_loss)
        print(f"Validation accuracy: {val_accuracy_epoch:.4f}, Validation loss: {avg_validation_loss:.4f}")

        if val_accuracy_epoch > last_validation_accuracy:
            last_validation_accuracy = val_accuracy_epoch
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cycles:
                print(f"Training stopped after {cycles} epochs with no improvement.")
                break

    if best_model is not None:
        model.load_state_dict(best_model)

    # Test phase
    model.eval()
    correct = 0
    total = 0
    print("========== Testing started ==========")
    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words).translate(str.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding else word_embedding['unk'] for i in input_words]
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        correct += int(predicted_label == gold_label)
        total += 1
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Plotting
    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('rnn1_accuracy_curves.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('rnn1_loss_curves.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig('rnn1_learning_curve.png')
    plt.show()
