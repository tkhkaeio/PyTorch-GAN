from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #in,out, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(1, 16, 3, 1) #16->14->7  #28->24->12
        self.conv2 = nn.Conv2d(16, 40, 2, 1) #7->6->        #12->8->4
        self.fc1 = nn.Linear(3*3*40, 400) 
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3*3*40)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == args.log_interval-1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    target_names = ['class %d'%i for i in range(10)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            #print(classification_report(target, pred, target_names=target_names))
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100.*correct/len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--aug_datapath', default=None,
                        help='augmentation dataset path')
    parser.add_argument('--theta', type=int, default=15,
                        help='random roatation angle')
    parser.add_argument('--n_sample', type=int, default=10,
                        help='random roatation angle')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    #aug_datapath = "dcgan_outputs49" #"outputs99"
    try:
        train_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset((
                    datasets.ImageFolder("../../../data/mnist/train",
                        transform=transforms.Compose([
                            transforms.Grayscale(),
                            #transforms.RandomRotation((-args.theta,args.theta)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                    ])),
                    datasets.ImageFolder(args.aug_datapath,
                        transform=transforms.Compose([
                            transforms.Grayscale(),
                            #transforms.RandomRotation((-args.theta,args.theta)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                    ]))
                    )), batch_size=args.batch_size, shuffle=True, **kwargs)
        print(args.aug_datapath, " augmented")
    except:
        train_loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder("../../../data/mnist/train",
                        transform=transforms.Compose([
                            transforms.Grayscale(),
                            #transforms.RandomRotation((-args.theta,args.theta)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                    ])), batch_size=args.batch_size, shuffle=True, **kwargs)


    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder("../../../data/mnist/test",
                    transform=transforms.Compose([
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                ])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)

    TestLoss = []
    TestAcc = []
    for i in range(args.n_sample):
        print("sample ", i)
        args.seed += 1
        args.lr = 0.01
        print(args)
        torch.manual_seed(args.seed)
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test_loss, acc = test(args, model, device, test_loader)
            if epoch == 3:
                args.lr /= 10
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        TestLoss.append(test_loss)
        TestAcc.append(acc)
    print("{} epochs done: test loss: {:.3f}\pm{:.3f}, acc: {:.4f}\pm{:.4f}".format(args.n_sample, np.mean(TestLoss), 1.96*np.std(TestLoss)/np.sqrt(args.n_sample), np.mean(TestAcc), 1.96*np.std(TestAcc)/np.sqrt(args.n_sample)))

    if(args.save_model): torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()