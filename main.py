# import
from os.path import basename, join
from glob import glob
import torch
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from src.model import *
from src.train import *
from src.evaluation import *
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt


#global parameters
USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 1234

# def


def load_data(data_path, sample_rate):
    files_path = sorted(glob(join(data_path, '*.wav')))
    waves = []
    label = []
    max_length = 0
    for path in files_path:
        wav, _ = librosa.load(path, sample_rate)
        max_length = len(wav) if len(wav) > max_length else max_length
        label.append(int(basename(path)[:-4])//100)
        waves.append(wav)
    label = np.array(label)
    return waves, label, max_length


def wav2mfcc(waves, sample_rate, max_length):
    data = []
    for wav in waves:
        diff = max_length-len(wav)
        if diff:
            wav = np.append(wav, np.zeros(diff, np.float32))
        elif diff < 0:
            wav = wav[:max_length]
        data.append(librosa.feature.mfcc(wav, sample_rate))
    return np.array(data)


if __name__ == "__main__":
    # parameters
    data_path = './data/speaker1'
    sample_rate = 16000
    batch_size = 10
    lr = 0.0001
    epochs = 1000
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # load data
    waves, label, max_length = load_data(data_path, sample_rate)

    # feature extraction
    data = wav2mfcc(waves, sample_rate, max_length)

    # prepare dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, label, test_size=0.3)
    train_set = TensorDataset(torch.from_numpy(
        X_train).float(), torch.from_numpy(Y_train))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_set = TensorDataset(torch.from_numpy(
        X_test).float(), torch.from_numpy(Y_test))
    test_loader = DataLoader(dataset=test_set, num_workers=4, pin_memory=True)

    # create model
    deepcnn = DEEPCNN(in_channels=data.shape[1], kernel_size=3,
                      in_dim=data.shape[-1], out_dim=10, hidden_dim=100, n_hidden=3, dropout=0.5)
    optimizer = optim.Adam(deepcnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        deepcnn = deepcnn.cuda()

    # train
    history = train_loop((train_loader, test_loader), deepcnn,
                         optimizer, criterion, epochs)

    # evaluation
    train_pred, train_acc, train_prob = evaluation(train_loader, deepcnn)
    print('Train dataset accuracy: {}'.format(round(np.mean(train_acc), 4)))
    test_pred, test_acc, test_prob = evaluation(test_loader, deepcnn)
    print('Test dataset accuracy: {}'.format(round(np.mean(test_acc), 4)))

    # predict other speaker
    unsee_waves, unsee_label, _ = load_data('./data/speaker2', sample_rate)
    unsee_data = wav2mfcc(unsee_waves, sample_rate, max_length)
    unsee_set = TensorDataset(torch.from_numpy(
        unsee_data).float(), torch.from_numpy(unsee_label))
    unsee_loader = DataLoader(
        dataset=unsee_set, num_workers=4, pin_memory=True)
    unsee_pred, unsee_acc, unsee_prob = evaluation(unsee_loader, deepcnn)
    print('Unsee dataset accuracy: {}'.format(round(np.mean(unsee_acc), 4)))
