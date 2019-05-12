from dataset import read_data, get_train_test
from models.custom_cnn import Net
from utils import splitsongs, to_melspectrogram
import torch.optim as optim
import torch
import os
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, required=True,
                    help='root directory for the dataset')
parser.add_argument('--epochs', type=int, default=100,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay for L2 penalty')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(X_train, y_train, batch_size, epochs):

    X_train = torch.tensor(X_train, device = device)
    y_train = torch.tensor(y_train, device = device)
    net = Net()
    net = net.to(device)
    net = net.type(torch.cuda.DoubleTensor)
    net.train();

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        iters = X_train.shape[0]//batch_size + 1

        for i in range(1, iters):
            # get the inputs
            inputs = X_train[(i-1)*batch_size:i*batch_size, :, :]
            labels = y_train[(i-1)*batch_size:i*batch_size]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:    # print every 2000 mini-batches
                print('epoch: [%d] loss: %.3f' %
                      (epoch + 1, running_loss / 400))
                running_loss = 0.0

    print('Finished Training')
    return net

def test(X_test, y_test, model):

    X_test = torch.tensor(X_test, device = device)
    y_test = torch.tensor(y_test, device = device)
    correct = 0
    total = 0
    model.eval();
    with torch.no_grad():
        size = X_test.shape[0]
        for i in range(size):
            image = X_test[i:i+1,:,:,:]
            label = y_test[i]
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == label).sum().item()

    return 100 * correct / total

def main():

    gtzan_dir = args.root_dir + '/genres/'
    song_samples = 660000
    genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
              'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

    # Read the data
    print("Reading in the data..")
    if os.path.isfile(os.path.join(args.root_dir, "x_gtzan_npy.npy")) and os.path.isfile(os.path.join(args.root_dir, "y_gtzan_npy.npy")):
        X = np.load("x_gtzan_npy.npy")
        y = np.load("y_gtzan_npy.npy")
    else:
        X, y = read_data(gtzan_dir, genres, song_samples, to_melspectrogram, debug=False)
        np.save('x_gtzan_npy.npy', X)
        np.save('y_gtzan_npy.npy', y)
    print("Completed reading the data!")

    X_train, X_test, y_train, y_test = get_train_test(X, y)

    print("Training the model..")
    model = train(X_train, y_train, args.batch_size, args.epochs)
    print("Training completed!")
    print("Doing the inference on the test set..")
    test_acc = test(X_test, y_test, model)
    print('Accuracy of the network on the 5240 test images: %d %%' % (test_acc))

if __name__ == "__main__":
    main()
