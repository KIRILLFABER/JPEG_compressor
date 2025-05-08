import func
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import *
import math


def compress(im):
    y, Cb, Cr = func.RGB_to_YCbCr(im)
    Cb = func.downsampling(Cb)
    Cr = func.downsampling(Cr)
    y_blocks = func.split_into_blocks(y)
    Cb_blocks = func.split_into_blocks(Cb)
    Cr_blocks = func.split_into_blocks(Cr)
    comp_data = bytearray()
    h,w,c = im.shape # перевести в байты для изображений боольше 255х255
    comp_data.append(h) 
    comp_data.append(w)
    comp_data.append(c)
    bitstream = ''
    for blocks, Q, channel in [(y_blocks, Q_Y, 'y'), (Cb_blocks, Q_C, 'Cb'), (Cr_blocks, Q_C, 'Cr')]:
        prev_dc = {'y' : 0, 'Cb': 0, 'Cr': 0}
        ha_dc_table = {'y' : HA_Y_DC_TABLE, 'Cb': HA_C_DC_TABLE, 'Cr': HA_C_DC_TABLE}
        ha_ac_table = {'y' : HA_Y_AC_TABLE, 'Cb': HA_C_AC_TABLE, 'Cr': HA_C_AC_TABLE}
        for row in blocks: 
            for block in row:
                dct_matrix = func.dct_2d(block - 128)
                q_matrix = func.quantize(dct_matrix, Q)
                zz = func.zigzag(q_matrix)
                dc_diff = zz[0] - prev_dc[channel]
                prev_dc[channel] = zz[0]
                dc_size = 0 if dc_diff == 0 else int(math.log2(abs(dc_diff))) + 1
                # dc_code = ha_dc_table[channel].get(dc_size, '')
                dc_code = ha_dc_table[channel][dc_size]
                if dc_size > 0:
                    if dc_diff > 0:
                        dc_code += bin(dc_diff)[2:].zfill(dc_size)
                    else:
                        dc_code += bin((1 << dc_size) + dc_diff - 1)[2:]
                ac_rle = func.rle_ac(zz[1:])
                ac_bits = func.huffman_encode_ac(ac_rle, ha_ac_table[channel])
                # if blocks == y_blocks:
                #     print(ac, '\t', func.huffman_encode_ac(ac, HA_Y_AC_TABLE))
                full_bits = dc_code + ac_bits
                bitstream += full_bits
    packed_data, pad_len = func.pack_to_bytes(bitstream)
    comp_data.extend(packed_data)
    comp_data.append(pad_len)
    return comp_data




            
            
        

def decompress(data): 
    orig_h = data[0]
    orig_w = data[1]
    c = data[2]
    data = data[3:]

    padded_h = math.ceil(orig_h / 8) * 8
    padded_w = math.ceil(orig_w / 8) * 8
    y = np.zeros((padded_h, padded_w)).astype(np.uint8)
    Cb = np.zeros((padded_h, padded_w))
    Cr = np.zeros((padded_h, padded_w))
    pad_len = data[-1]
    data = data[:-1]
    bitstream = ''.join(f'{byte:08b}' for byte in data)
    if pad_len != 0:
        bitstream = bitstream[:-pad_len]    
    #print(len(bitstream))
    #print(padded_h, padded_w)
    current_pos = 0
    c = 0
    prev_dc = 0
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            #print(f"Decoding block ({i}, {j}), bit pos: {current_pos}")
            if i == 160 and j == 72:
                print(bitstream[current_pos:current_pos+20])
            dc_coeff, bits_consumed = func.huffman_decode_dc(bitstream[current_pos:], HA_Y_DC_TABLE)
            dc_coeff += prev_dc
            prev_dc = dc_coeff
            #print(dc_coeff)
            current_pos += bits_consumed
            if current_pos >= len(bitstream):
                break
            
            ac_coeffs, bits_used = func.huffman_decode_ac(bitstream[current_pos:], HA_Y_AC_TABLE)
            current_pos += bits_used
            if current_pos >= len(bitstream):
                break

            ac_coeffs = func.irle_ac(ac_coeffs)
            zz = [dc_coeff] + ac_coeffs
            quantized_block = func.inverse_zigzag(np.array(zz))
            
            block = (func.idct_2d(func.dequantize(quantized_block, Q_Y)) + 128).astype(np.uint8)
            print(c)
            c+=1
            y[i:i+8, j:j+8] = block
    
    
    return y




im_RGB = Image.open("data/Lenna.png")
size = im_RGB.size
im_RGB = np.array(im_RGB)
c = compress(im_RGB)
d = decompress(c)
print(d)
tmp_chanel = Image.new("L", size, 128)
y = Image.fromarray(d)
fig, ax = plt.subplots()
#im_y = Image.merge("YCbCr", (tmp_chanel, tmp_chanel, Cr)).convert('RGB')
ax.imshow(y)
plt.show()
    





    