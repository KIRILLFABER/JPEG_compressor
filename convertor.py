import func
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im_RGB = Image.open("data/Lenna.png")
im_GS = im_RGB.convert("L")
im_D  = im_RGB.convert("1")
im_WD  = im_RGB.convert("1", dither = Image.Dither.NONE)
im_GS.save('data/Lenna_GS.png')
im_D.save('data/Lenna_D.png')
im_WD.save('data/Lenna_WD.png')



