from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def getRGB(im):
    return im[:, :, 0].astype(np.float32), im[:, :, 1].astype(np.float32), im[:, :, 2].astype(np.float32)

def RGB_to_YCbCr(im):
    r, g, b = getRGB(im)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g - 0.0813 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    y = np.clip(y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    return y, Cb, Cr


def downsampling(chanel, factor = 2):
    h, w = chanel.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    padded = np.pad(chanel, ((0, pad_h), (0, pad_w)) if (pad_h or pad_w) else chanel)
    new_h, new_w = padded.shape
    
    # Разбиваем на блоки 4x4
    blocks = padded.reshape(new_h // 4, 4, new_w // 4, 4)
    blocks = blocks.transpose(0, 2, 1, 3)  # Перегруппировка осей
    # print(chanel)
    # print('===============')
    # print(blocks)
    downsampled = blocks.mean(axis=(2, 3)).astype(np.uint8)  # Усреднение по каждому блоку
    # print(chanel.shape)
    # print(downsampled.shape)
    return downsampled

import numpy as np
import math

def split_into_blocks(channel, block_size=8, fill_value=0):
    """
    Разбивает один канал (Y, Cb или Cr) на блоки N×N.
    
    Args:
        channel: 2D numpy array (один канал изображения).
        block_size: размер блока (по умолчанию 8).
        fill_value: значение для заполнения неполных блоков.
    
    Returns:
        Список списков блоков (в виде 2D numpy массивов).
    """
    h, w = channel.shape
    # Вычисляем количество блоков (округляем вверх)
    n_blocks_h = math.ceil(h / block_size)
    n_blocks_w = math.ceil(w / block_size)
    
    # Создаем массив с заполнением (pad до кратности block_size)
    padded_h = n_blocks_h * block_size
    padded_w = n_blocks_w * block_size
    padded_channel = np.full((padded_h, padded_w), fill_value, dtype=channel.dtype)
    padded_channel[:h, :w] = channel  # Копируем исходные данные
    
    # Разбиваем на блоки
    blocks = []
    for i in range(n_blocks_h):
        row_blocks = []
        for j in range(n_blocks_w):
            block = padded_channel[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size
            ]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return blocks




import numpy as np
import math

def dct_1d(x):
    """1D DCT-II для вектора x"""
    N = len(x)
    X = np.zeros(N)
    
    for k in range(N):
        sum_val = 0.0
        ck = 1.0 / math.sqrt(2) if k == 0 else 1.0
        
        for n in range(N):
            angle = (math.pi / N) * (n + 0.5) * k
            sum_val += x[n] * math.cos(angle)
        
        X[k] = math.sqrt(2.0 / N) * ck * sum_val
    
    return X

def idct_1d(X):
    """1D IDCT-II для вектора X"""
    N = len(X)
    x = np.zeros(N)
    
    for n in range(N):
        sum_val = 0.0
        
        for k in range(N):
            ck = 1.0 / math.sqrt(2) if k == 0 else 1.0
            angle = (math.pi / N) * (n + 0.5) * k
            sum_val += ck * X[k] * math.cos(angle)
        
        x[n] = math.sqrt(2.0 / N) * sum_val
    
    return x

def dct_2d(block):
    """2D DCT-II для блока N x N"""
    N = block.shape[0]
    dct_block = np.zeros((N, N))
    
    # Применяем 1D DCT к каждой строке
    for i in range(N):
        dct_block[i, :] = dct_1d(block[i, :])
    
    # Применяем 1D DCT к каждому столбцу
    for j in range(N):
        dct_block[:, j] = dct_1d(dct_block[:, j])
    
    return dct_block

def idct_2d(dct_block):
    """2D IDCT-II для блока N x N"""
    N = dct_block.shape[0]
    block = np.zeros((N, N))
    
    # Применяем 1D IDCT к каждому столбцу
    for j in range(N):
        block[:, j] = idct_1d(dct_block[:, j])
    
    # Применяем 1D IDCT к каждой строке
    for i in range(N):
        block[i, :] = idct_1d(block[i, :])
    
    return block





def dct(matrix):
    dct_matrix = np.zeros(matrix.shape)



def quantize(block, q_matrix, quality=50):
    """Квантование с учетом качества"""
    if quality <= 0:
        quality = 1
    elif quality > 100:
        quality = 100
    
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.floor((q_matrix * scale + 50) / 100).clip(1, 255)
    return np.round(block / q).astype(np.int32)

def zigzag(block):
    """Зигзаг-сканирование"""
    return np.concatenate([np.diagonal(block[::-1,:], i)[::(2*(i % 2)-1)] 
                         for i in range(1-block.shape[0], block.shape[0])])

def differential_encode(dc_values):
    """Разностное кодирование DC коэффициентов"""
    return [dc_values[0]] + [dc_values[i] - dc_values[i-1] for i in range(1, len(dc_values))]

def rle_ac(ac_coeffs):
    """RLE кодирование AC коэффициентов"""
    rle = []
    zero_run = 0
    for coeff in ac_coeffs:
        if coeff == 0:
            zero_run += 1
        else:
            while zero_run > 15:
                rle.append((15, 0))
                zero_run -= 16
            rle.append((zero_run, coeff))
            zero_run = 0
    rle.append((0, 0))  # EOB
    return rle



# def quantize(dct_block, q_matrix):
#     """Квантование DCT коэффициентов"""
#     return np.round(dct_block / q_matrix).astype(np.int32)
# def dequantize(quantized_block, q_matrix):
#     """Обратное квантование"""
#     return quantized_block * q_matrix


def zigzag_order(block_size=8):
    """Генерация зигзаг-последовательности для блока NxN"""
    zigzag = []
    for s in range(2 * block_size - 1):
        if s < block_size:
            if s % 2 == 0:
                i, j = s, 0
                while i >= 0:
                    zigzag.append((i, j))
                    i -= 1
                    j += 1
            else:
                i, j = 0, s
                while j >= 0:
                    zigzag.append((i, j))
                    i += 1
                    j -= 1
        else:
            if s % 2 == 0:
                i, j = block_size-1, s - block_size + 1
                while j < block_size:
                    zigzag.append((i, j))
                    i -= 1
                    j += 1
            else:
                i, j = s - block_size + 1, block_size-1
                while i < block_size:
                    zigzag.append((i, j))
                    i += 1
                    j -= 1
    return zigzag

def zigzag_scan(block):
    """Зигзаг-сканирование блока"""
    block_size = block.shape[0]
    order = zigzag_order(block_size)
    return np.array([block[i, j] for (i, j) in order])

def inverse_zigzag(zigzag_data, block_size=8):
    """Обратное зигзаг-сканирование"""
    block = np.zeros((block_size, block_size), dtype=zigzag_data.dtype)
    order = zigzag_order(block_size)
    
    for idx, (i, j) in enumerate(order):
        if idx < len(zigzag_data):
            block[i, j] = zigzag_data[idx]
    
    return block







def zigzag():
    pass





im_RGB = Image.open("1.jpg")
size = im_RGB.size
im_RGB = np.array(im_RGB)
y, Cb, Cr = RGB_to_YCbCr(im_RGB)




# y, Cb, Cr = RGB_to_YCbCr(im_RGB)
# print(im_RGB)
# r, g, b = getRGB(im_RGB)
# fig, ax = plt.subplots()
# print('==========', r)





# tmp_chanel = Image.new("L", size, 128)
# y = Image.fromarray(y)
# Cb = Image.fromarray(Cb)
# Cr = Image.fromarray(Cr)
# im_y = Image.merge("YCbCr", (tmp_chanel, tmp_chanel, Cr)).convert('RGB')
# ax.imshow(im_y)
# plt.show()
