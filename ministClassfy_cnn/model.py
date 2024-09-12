from torch import nn
from torch.nn import functional as F

class MINISTCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2), # 16*28*28
            nn.ReLU(),
            nn.MaxPool2d(2)         ## 16*14*214
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.Conv2d(32,32,5,1,2), ## 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)         ## 32*7*7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),              #64*7*7
        )
       
        self.out = nn.Linear(64*7*7,10) # 
       
    
    # torch 中 前向传播需要自定义
    # 反向传播自动进行
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0),-1) # 把(b,c,h,w) -> (b,c*h*w)
    
        output = self.out(x)
        return output

