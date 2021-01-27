import torch
import torch.optim as optim
from DivGraphPointer import *

def train_divnet(n_epochs, data_obj):

    #torch.autograd.set_detect_anomaly(True)
    diven = DivGraphEncoder(100, 100, 1)
    divdec = DivGraphDecoder(100, 300, 1)
    divnet = DivGraphNet(diven, divdec)

    optimiser = optim.Adam(divnet.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        optimiser.zero_grad()
        divnet.train()
        kp, loss = divnet(data_obj, data_obj.labels)
        loss.backward()
        optimiser.step()
        print('Loss for epoch', epoch)
        print(loss)

    return kp
