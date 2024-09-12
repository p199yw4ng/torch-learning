from torch import nn
from torch.nn import functional as F

class MINIST(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden1 = nn.Linear(784,128) # 输入784像素点 输出128特征
        self.hidden2 = nn.Linear(128,256) # 输入128特征 输出256特征
        # self.hidden3 = nn.Linear(256,512)
        self.out = nn.Linear(256,10) # 
        self.dropout = nn.Dropout(0.5) # 降低过拟合，随机丢弃,0.5比较常见
    
    # torch 中 前向传播需要自定义
    # 反向传播自动进行
    def forward(self,x):
        x = F.relu(self.hidden1(x)) 
        x = self.dropout(x) # 一般全连接层都需要加上dropout
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

