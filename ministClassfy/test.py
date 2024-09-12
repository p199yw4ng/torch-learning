from PIL import Image
import numpy as np
img = np.asarray(Image.open("1.png"))
print(img.flatten().shape)