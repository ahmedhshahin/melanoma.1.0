import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from pytorch1 import melanomaData
from unet import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.51892472, 0.4431646,  0.40640972], [0.37666158, 0.33505249, 0.32253156])])
# Call the dataset
dset_train = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=True)
dset_val = melanomaData('/home/ahmed/github/melanoma.1.0/dataset/2016data/train/', transformations, is_train=False)
train_dloader = DataLoader(dset_train, batch_size=4, shuffle=True, num_workers=4)
val_dloader = DataLoader(dset_val, batch_size=1, shuffle=True, num_workers=4)

N_train = len(train_dloader)
def train(net, epochs=5, batch_size=2, lr=0.1):
	
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators
        epoch_loss = 0
        i = 0
        for img , label in train_dloader:
            img = Variable(img)#.cuda()
            label = Variable(label)#.cuda()
            
            y_pred = net(img)
            probs = F.sigmoid(y_pred)
            
            probs_flat = probs.view(-1)
            y_flat = label.view(-1)

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.data[0]

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train,
                                                     loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            i += 1

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

net = UNet(3, 1)
train(net, batch_size=1)