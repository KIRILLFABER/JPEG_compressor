import func
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


path = './data/'
files = [path + 'Lenna', path + 'img']
for file in files:
    im_RGB = Image.open(file + '.png')
    im_GS = im_RGB.convert("L")
    im_D  = im_RGB.convert("1")
    im_WD  = im_RGB.convert("1", dither = Image.Dither.NONE)
    im_GS.save(file + '_GS.png')
    im_D.save(file + '_D.png')
    im_WD.save(file + '_WD.png')








