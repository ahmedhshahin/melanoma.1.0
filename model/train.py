import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from pytorch1 import melanomaData
from unet import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])
# Call the dataset
dset_train = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=True)
dset_val = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=False)
train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)

N_train = len(train_dloader)

def get_acc(model, loader):
    n_correct = 0
    n_samples = 0
    model.eval()
    for x, y in loader:
        x_var = Variable(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    return acc

def train(net, epochs=5, batch_size=2, lr=0.1):
	
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators
        epoch_loss = 0
        for i, (img , label) in enumerate(train_dloader):
            img = Variable(img)#.cuda()
            label = Variable(label)#.cuda()
            
            y_pred = net(img)
            probs = F.sigmoid(y_pred)
            
            probs_flat = probs.view(-1)
            y_flat = label.view(-1)

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.item()

            print('{0} / {1:d} --- loss: {2:.6f}'.format(i, math.ceil(N_train / batch_size),
                                                     loss.item()))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        print('Epoch finished ! Loss: {0} - Training Accuracy: {1}'.format(epoch_loss / i, get_acc(net, train_dloader)))

net = UNet(3, 1)
train(net, batch_size=1)