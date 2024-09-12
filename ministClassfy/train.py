from model import MINIST
from torch.utils.data import DataLoader
from torch.nn import functional as F
from ministDataset import MINISTDATASET
from tqdm import tqdm
import torch
import os
from torch import optim

PATH = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    device = 'cuda:0'
    epochs = 20
    # 模型
    model = MINIST()
    # 优化器
    opt = optim.Adam(model.parameters(),lr=0.001) 
    # loss func
    crossloss = F.cross_entropy
    model.to(device)
    # 加载数据
    train_dataset = MINISTDATASET(os.path.join(PATH,'MNIST/train'))
    val_dataset = MINISTDATASET(os.path.join(PATH,'MNIST/val'))
    train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=256,shuffle=True,num_workers=4)
    
    
    # train
    best_loss = 9999
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        for imgs,labels in tqdm(train_loader):
            imgs = imgs.to(device)      
            labels = labels.to(device)
            pred = model(imgs)
            #反向传播
            loss = crossloss(pred,labels)
            train_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        train_loss = train_loss/len(train_dataset)
        print(f'Epoch:{epoch}\t Train Loss:{train_loss}')
        if best_loss>train_loss:
            best_loss = train_loss
            
        

        model.eval()
        with torch.no_grad():
            correct=0
            for imgs,labels in tqdm(val_loader):
                imgs = imgs.to(device)      
                labels = labels.to(device)
                pred = model(imgs)
                _,predicted = torch.max(pred,1)
                correct+=(predicted==labels).sum().item()
                #反向传播

            print(f'Epoch:{epoch}\t ACC:{correct/len(val_dataset)}%')
    torch.save(model.state_dict(), 'best_model.pth')