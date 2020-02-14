# import
import numpy as np
import torch
from sklearn.metrics import accuracy_score

#global parameters
USE_CUDA = torch.cuda.is_available()

# def


def evaluation(dataloader, model):
    predict = []
    accuracy = []
    probability = []
    model.eval()
    for x, y in dataloader:
        if USE_CUDA:
            x, y = x.cuda(), y.cuda()
        prob = model(x).cpu().data.numpy()
        pred = np.argmax(prob, 1)
        acc = accuracy_score(y.cpu().data.numpy(), pred)
        predict.append(pred)
        accuracy.append(acc)
        probability.append(prob)
    return predict, accuracy, probability
