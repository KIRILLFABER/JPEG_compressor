import func
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


PATH = './data/'


def to_raw(im, name):
    im = np.array(im).flatten()
    raw_path = './data/raw_data/'
    f = open(raw_path + name, 'wb')
    f.write(im)
    f.close()

files = [PATH + 'Lenna.png', PATH + 'img.png']
for file in files:
    im_RGB = Image.open(file)
    im_GS = im_RGB.convert("L")
    im_D  = im_RGB.convert("1")
    im_WD  = im_RGB.convert("1", dither = Image.Dither.NONE)
    im_GS.save(file[:-4] + '_GS.png')
    im_D.save(file[:-4] + '_D.png')
    im_WD.save(file[:-4] + '_WD.png')
    to_raw(im_RGB, file[len(PATH):-4] + 'RGB')
    to_raw(im_GS, file[len(PATH):-4] + 'GS')
    to_raw(im_D, file[len(PATH):-4] + 'D')
    to_raw(im_WD, file[len(PATH):-4] + 'WD')










