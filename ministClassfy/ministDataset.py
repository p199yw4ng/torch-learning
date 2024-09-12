
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MINISTDATASET(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.lables=['0','1','2','3','4','5','6','7','8','9']
        self.itmes=[]
        for lable in self.lables:
            for file in os.listdir(os.path.join(path,lable)):
                self.itmes.append({
                    'filename':os.path.join(path,lable,file),
                    'lable':int(lable)
                })

    def __getitem__(self, index):
        filename = self.itmes[index]['filename']
        img = np.asarray(Image.open(filename)).astype(np.float32)
        lable = self.itmes[index]['lable']
        img = img.flatten()
        return img,lable
    def __len__(self):
        return len(self.itmes)