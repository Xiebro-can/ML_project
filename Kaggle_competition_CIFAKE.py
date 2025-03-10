import torch
import torch.utils
import torch.utils.data
import torchvision as tv
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,random_split

#image augmentation
transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0),(1,1,1))
])

#load data
train_dataset=ImageFolder(root='train',transform=transform)
test_dataset=ImageFolder(root='test',transform=transform)
device=torch.device('cuda:0')
train_len=int(0.8*len(train_dataset))
val_len=len(train_dataset)-train_len
train_set,val_set=torch.utils.data.random_split(train_dataset,[train_len,val_len])
train_loader=DataLoader(train_set,batch_size=512,shuffle=True,num_workers=4)
val_loader=DataLoader(val_set,batch_size=512,shuffle=False,num_workers=4)
test_loader=DataLoader(test_dataset,batch_size=512,shuffle=False,num_workers=4)

#resnet
class resnet(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
        self.conv1=nn.Conv2d(input_channel,output_channel,3,padding=1,stride=1)
        self.conv2=nn.Conv2d(output_channel,output_channel,3,padding=1,stride=1)
        self.conv3=nn.Conv2d(input_channel,output_channel,kernel_size=1,padding=0)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.bn2=nn.BatchNorm2d(output_channel)
    def forward(self,X):
        Y=self.conv3(X)
        X=self.conv1(X)
        X=self.bn1(X)
        X=self.conv2(torch.relu(X))
        X=self.bn2(X)
        Y+=X
        return Y

#DIY resnet
net=nn.Sequential(
    resnet(3,16),
    nn.MaxPool2d(2,2),
    resnet(16,64),
    nn.MaxPool2d(2,2),
    resnet(64,128),
    nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(2048,1024),nn.ReLU(),
    nn.Linear(1024,128),nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128,32),nn.Sigmoid(),
    nn.Linear(32,2)
)

#initial weight
def initweight(m):
    if m==nn.Linear or m==nn.Conv2d:
        nn.init.xavier_normal_(m)

#hyper-parameter
lr=0.001
num_epochs=30

#main function
if __name__=='__main__':
    best_accuracy=0
    net.load_state_dict(torch.load('best_model.pth'))  #fine-tuning
    #net.apply(initweight)                             if you want to retraining
    net=net.to(device)
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr)
    for epoch in range(num_epochs):
        net.train()
        runloss=0.0
        total=0.0
        right=0.0
        for X,y in train_loader:
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            l=loss(net(X),y).sum()
            runloss+=l
            l.backward()
            optimizer.step()
        print('epoch',epoch+1,'loss',runloss)
        net.eval()
        with torch.no_grad():
            for X,y in val_loader:
                X,y=X.to(device),y.to(device)
                y_hat=net(X)
                _,predicts=torch.max(y_hat,1)
                total+=y.size(0)
                right+=(predicts==y).sum().item()
        train_acc=100*right/total
        print('train_acc:',train_acc)
        test_total=0
        test_right=0
        with torch.no_grad():
                for X,y in test_loader:
                    X,y=X.to(device),y.to(device)
                    y_hat=net(X)
                    _,predicts=torch.max(y_hat,1)
                    test_total+=y.size(0)
                    test_right+=(predicts==y).sum().item()
        test_acc=100*test_right/test_total
        print('test_acc:',test_acc)
        #save the best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(net.state_dict(), 'best_model.pth')



