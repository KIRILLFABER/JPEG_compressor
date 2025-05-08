from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def getRGB(im):
    return im[:, :, 0].astype(np.float32), im[:, :, 1].astype(np.float32), im[:, :, 2].astype(np.float32)

def RGB_to_YCbCr(im):
    r, g, b = getRGB(im)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g - 0.0813 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    y = np.clip(y, 0, 255).astype(np.int32)
    Cb = np.clip(Cb, 0, 255).astype(np.int32)
    Cr = np.clip(Cr, 0, 255).astype(np.int32)
    return y, Cb, Cr


def downsampling(chanel, factor = 2):
    h, w = chanel.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        padded = np.pad(chanel, ((0, pad_h), (0, pad_w)), mode='edge')
    else:
        padded = chanel

    new_h, new_w = padded.shape
    
    blocks = padded.reshape(new_h // 4, 4, new_w // 4, 4)
    blocks = blocks.transpose(0, 2, 1, 3)
    # print(chanel)
    # print('===============')
    # print(blocks)
    downsampled = blocks.mean(axis=(2, 3)).astype(np.int32)
    # print(chanel.shape)
    # print(downsampled.shape)
    return downsampled


def split_into_blocks(channel, block_size=8, fill_value=0):
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
            block = padded_channel[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size
            ]
            row_blocks.append(block)
        blocks.append(row_blocks)
    
    return blocks




def dct_1d(x):
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
    N = block.shape[0]
    dct_block = np.zeros((N, N))
    for i in range(N):
        dct_block[i, :] = dct_1d(block[i, :])
    for j in range(N):
        dct_block[:, j] = dct_1d(dct_block[:, j])
    
    return dct_block

def idct_2d(dct_block):
    N = dct_block.shape[0]
    block = np.zeros((N, N))
    for j in range(N):
        block[:, j] = idct_1d(dct_block[:, j])
    for i in range(N):
        block[i, :] = idct_1d(block[i, :])
    
    return block











def quantize(block, q_matrix, quality=50):
    if quality <= 0:
        quality = 1
    elif quality > 100:
        quality = 100
    
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.floor((q_matrix * scale + 50) / 100).clip(1, 255)
    return np.round(block / q).astype(np.int32)

def dequantize(block, q_matrix, quality=50):
    if quality <= 0:
        quality = 1
    elif quality > 100:
        quality = 100

    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    q = np.floor((q_matrix * scale + 50) / 100).clip(1, 255)
    return (block * q).astype(np.int32)


def zigzag(block):
    return np.concatenate([np.diagonal(block[::-1,:], i)[::(2*(i % 2)-1)] 
                         for i in range(1-block.shape[0], block.shape[0])])


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
    return [flat_block[i] for i in zigzag_order]


def calculate_size(value):
    if value == 0:
        return 0
    return value.bit_length()

def huffman_encode_ac(rle_pairs, huffman_table):
    bitstream = ""
    for run, value in rle_pairs:
        if (run, value) == (0, 0):  # EOB
            bitstream += huffman_table[(0, 0)]
            break
        size = 0 if value == 0 else max(1, int(math.log2(abs(value))) + 1)
        code = huffman_table[(run, size)]
        bitstream += code
        if value != 0:
            if value > 0:
                bitstream += bin(value)[2:].zfill(size)
            else:
                bitstream += bin((1 << size) + value - 1)[2:]
    return bitstream

def huffman_decode_ac(bitstream, huffman_table, max_coeffs=63):
    inv_huffman_table = {v: k for k, v in huffman_table.items()}
    rle_pairs = []
    i = 0
    n = len(bitstream)
    bits_consumed = 0

    while i < n and len(rle_pairs) < max_coeffs:
        code = ""
        found = None
        while i < n and found is None:
            code += bitstream[i]
            i += 1
            found = inv_huffman_table.get(code)
        
        run, size = found
        bits_consumed += len(code) 

        if (run, size) == (0, 0):
            rle_pairs.append((0, 0))
            break
        value = 0
        if size > 0:
            if i + size > n:
                raise ValueError("Неверный битовый поток: недостаточно битов для значения")
            
            value_bits = bitstream[i:i+size]
            i += size
            bits_consumed += size

            if value_bits[0] == '1':
                value = int(value_bits, 2)
            else:
                value = int(value_bits, 2) - (1 << size) + 1

        rle_pairs.append((run, value))

    return rle_pairs, bits_consumed



def huffman_decode_dc(bitstream, huffman_table):
    inv_table = {v: k for k, v in huffman_table.items()}
    code = ''
    for i, bit in enumerate(bitstream):
        code += bit
        if code in inv_table:
            size = inv_table[code]
            if size == 0:
                return 0, i + 1
            value_bits = bitstream[i+1:i+1+size]
            if len(value_bits) < size:
                raise ValueError("Недостаточно битов для DC значения")
            value = int(value_bits, 2)
            if value_bits[0] == '0':
                value -= (1 << size) - 1
            return value, i + 1 + size
    raise ValueError("Invalid DC Huffman code")
    

def pack_to_bytes(bitstream):
    pad_len = (8 - len(bitstream) % 8) % 8
    bitstream += '0' * pad_len
    data = bytearray()
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i+8]
        data.append(int(byte, 2))
    
    return data, pad_len


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
