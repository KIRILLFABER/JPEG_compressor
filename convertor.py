import func
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def convertor(im, mode='grayscale'):
    y, Cb, Cr = func.RGB_to_YCbCr(np.array(im))
    tmp_chanel = Image.new("L", im.size, 128)
    y = Image.fromarray(y)
    result = Image.merge("YCbCr", (y, tmp_chanel, tmp_chanel)).convert('RGB')
    return result

path = './data/'
im = Image.open(path + 'Lenna.png')
g_im = convertor(im)
fig, ax = plt.subplots()
ax.imshow(g_im)
fig.savefig(path + 'g_' + 'Lenna.png')
plt.show


