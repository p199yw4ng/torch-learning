from model import MINISTCNN as MINIST
from torch.utils.data import DataLoader
import torch.nn  as nn
from ministDataset import MINISTDATASET
from tqdm import tqdm
import torch
import os
from torch import optim
from torchmetrics import Accuracy

PATH = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    batch_size = 256
    device = 'cuda:0'
    epochs = 2
    # 模型
    model = MINIST()
    # 优化器
    opt = optim.Adam(model.parameters(),lr=0.001) 
    # loss func
    crossloss = nn.CrossEntropyLoss()
    model.to(device)
    # 加载数据
    train_dataset = MINISTDATASET(os.path.join(PATH,'MNIST/train'))
    val_dataset = MINISTDATASET(os.path.join(PATH,'MNIST/val'))
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    
    # train

    for epoch in range(epochs):

        for idx,(imgs,labels) in enumerate(train_loader):
            model.train()
            imgs = imgs.to(device)      
            labels = labels.to(device)
            pred = model(imgs)
            #反向传播
            loss = crossloss(pred,labels)

            loss.backward()
            opt.step()
            opt.zero_grad()

            
  
            if idx % 100 == 0:
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

                print(f'Epoch:{epoch}\t [{idx*batch_size}/{len(train_dataset)}] test ACC:{correct/len(val_dataset)}%')
    torch.save(model.state_dict(), 'best_model.pth')