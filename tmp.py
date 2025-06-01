import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.fftpack as scp
from config import *
import struct

def getRGB(im):
    return im[:, :, 0].astype(np.float32), im[:, :, 1].astype(np.float32), im[:, :, 2].astype(np.float32)

def RGB_to_YCbCr(im):
    r, g, b = getRGB(im)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128


    y = np.clip(y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    
    return y, Cb, Cr


def YCbCr_to_RGB(y, Cb, Cr):
    y = y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    r = y + 1.402 * Cr
    g = y - 0.34414 * Cb - 0.71414 * Cr
    b = y + 1.772 * Cb
    
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    return r, g, b

def downsampling(chanel, factor = 4):
    pass



def upsampling(channel, factor=4):
    pass

def split_into_blocks(channel, block_size=8, fill_value=128):
    h, w = channel.shape
    n_blocks_h = math.ceil(h / block_size)
    n_blocks_w = math.ceil(w / block_size)
    padded_h = n_blocks_h * block_size
    padded_w = n_blocks_w * block_size
    padded_channel = np.full((padded_h, padded_w), fill_value, dtype=channel.dtype)
    padded_channel[:h, :w] = channel 
    blocks = []
    for i in range(n_blocks_h):
        row_blocks = []
        for j in range(n_blocks_w):
            block = padded_channel[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return blocks

def dct_2d_s(x):    
    return (DCT_MATRIX @ x) @ DCT_MATRIX.T
    return np.dot(np.dot(DCT_MATRIX, x), DCT_MATRIX.T)

def idct_2d_s(dct_block):
    return (DCT_MATRIX.T @ dct_block) @ DCT_MATRIX
    return np.dot(np.dot(DCT_MATRIX.T, dct_block), DCT_MATRIX)

                       


def quantize(block, q_matrix, quality=50):
    if quality <= 0:
        quality = 1
    elif quality >= 100:
        quality = 100
        return block
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.floor((q_matrix * scale + 50) / 100).clip(1, 255)
    quantized_block = np.round(block / q)
    return quantized_block

def dequantize(block, q_matrix, quality=50):
    if quality <= 0:
        quality = 1
    elif quality >= 100:
        quality = 100
        return block

    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.floor((q_matrix * scale + 50) / 100).clip(1, 255)
    dequantized_block = block * q
    return dequantized_block

def zigzag(block):
    zigzag_order = [
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
    flat_block = block.flatten()
    return np.array([flat_block[i] for i in zigzag_order])

def rle_ac(ac_coeffs):
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
    rle.append((0, 0)) 
    return rle

def irle_ac(ac_coeffs):
    decoded = []
    for run, value in ac_coeffs:
        if (run, value) == (0, 0):
            decoded.extend([0]*(63 - len(decoded)))
            break
        decoded.extend([0]*run)
        decoded.append(value)    
    return decoded[:63] 



def zigzag_order(block_size=8):
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
    block_size = block.shape[0]
    order = zigzag_order(block_size)
    return np.array([block[i, j] for (i, j) in order])

def inverse_zigzag(zigzag_data, block_size=8):
    block = np.zeros((block_size, block_size), dtype=zigzag_data.dtype)
    order = zigzag_order(block_size)
    
    for idx, (i, j) in enumerate(order):
        if idx < len(zigzag_data):
            block[i, j] = zigzag_data[idx]
    
    return block






def calculate_size(value):
    if value == 0:
        return 0
    return value.bit_length()


def huffman_encode_ac(rle_pairs, huffman_table):
    pass

def huffman_decode_ac(bitstream, huffman_table, max_coeffs=63):
    pass


def huffman_encode_dc(dc, prev_dc, huffman_table):
    pass

def huffman_decode_dc(bitstream, huffman_table):
    pass


def pack_to_bytes(bitstream):
    pad_len = (8 - len(bitstream) % 8) % 8
    bitstream += '0' * pad_len
    data = bytearray()
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i+8]
        data.append(int(byte, 2))
    
    return data, pad_len







def downsampling(channel_matrix):
    original_height, original_width = channel_matrix.shape

    new_height = math.ceil(original_height / 2)
    new_width = math.ceil(original_width / 2)

    downsampled_matrix = np.zeros((new_height, new_width), dtype=np.float32)

    for r_new in range(new_height):
        for c_new in range(new_width):
            r_orig_start = r_new * 2
            c_orig_start = c_new * 2

            block = channel_matrix[r_orig_start:min(r_orig_start + 2, original_height),
                    c_orig_start:min(c_orig_start + 2, original_width)]

            if block.size > 0:
                downsampled_matrix[r_new, c_new] = np.mean(block)
            else:
                downsampled_matrix[r_new, c_new] = 0

    return np.round(downsampled_matrix).astype(np.uint8)


def upsampling(channel_matrix, target_height, target_width):

    upsampled = channel_matrix.repeat(2, axis=0).repeat(2, axis=1)

    final_height = min(target_height, upsampled.shape[0])
    final_width = min(target_width, upsampled.shape[1])

    result = np.zeros((target_height, target_width), dtype=upsampled.dtype)
    result[:final_height, :final_width] = upsampled[:final_height, :final_width]

    return result