import func
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import *
import math
import struct


def compress(im, quality = 50):
    AC = []
    DC = []







    y, Cb, Cr = func.RGB_to_YCbCr(im)
    Cb = func.downsampling(Cb)
    Cr = func.downsampling(Cr)
    print(func.upsampling(Cb).shape)
    y_blocks = func.split_into_blocks(y)
    Cb_blocks = func.split_into_blocks(Cb)
    Cr_blocks = func.split_into_blocks(Cr)
    comp_data = bytearray()
    h,w,c = im.shape
    size = struct.pack('>HHH', h, w, c)
    comp_data.extend(size)
    print(size)
    bitstream = ''
    for blocks, Q, channel in [(y_blocks, Q_Y, 'y'), (Cb_blocks, Q_C, 'Cb'), (Cr_blocks, Q_C, 'Cr')]:
        prev_dc = {'y' : 0, 'Cb': 0, 'Cr': 0}
        ha_dc_table = {'y' : HA_Y_DC_TABLE, 'Cb': HA_C_DC_TABLE, 'Cr': HA_C_DC_TABLE}
        ha_ac_table = {'y' : HA_Y_AC_TABLE, 'Cb': HA_C_AC_TABLE, 'Cr': HA_C_AC_TABLE}
        for row in blocks: 
            for block in row:
                dct_matrix = func.dct_2d(block - 128)
                q_matrix = func.quantize(dct_matrix, Q, quality)
                zz = func.zigzag(q_matrix)
                prev = prev_dc[channel]
                DC.append(zz[0] - prev)
                dc_code, prev_dc[channel] = func.huffman_encode_dc(zz[0], prev, ha_dc_table[channel])
                ac_rle = func.rle_ac(zz[1:])
                ac_bits = func.huffman_encode_ac(ac_rle, ha_ac_table[channel])
                # if blocks == y_blocks:
                #     print(ac, '\t', func.huffman_encode_ac(ac, HA_Y_AC_TABLE))
                full_bits = dc_code + ac_bits
                bitstream += full_bits
                #print(dc_code)
                #print(func.huffman_decode_dc(dc_code, HA_Y_DC_TABLE))

    packed_data, pad_len = func.pack_to_bytes(bitstream)
    comp_data.extend(packed_data)
    comp_data.append(pad_len)
    return comp_data, DC




            
            
        

def decompress(data, quality = 50):
    DC = []

    orig_h, orig_w, orig_c = struct.unpack('>HHH', data[:6]) 
    data = data[6:]
    #print(orig_h, orig_w, orig_c)
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
            dc_coeff, bits_consumed = func.huffman_decode_dc(bitstream[current_pos:], HA_Y_DC_TABLE)
            DC.append(dc_coeff)
            dc_coeff += prev_dc
            prev_dc = dc_coeff
            #print(dc_coeff)
            current_pos += bits_consumed
            if current_pos >= len(bitstream):
                break
            
            ac_coeffs, bits_used = func.huffman_decode_ac(bitstream[current_pos:], HA_Y_AC_TABLE)
            current_pos += bits_used

            ac_coeffs = func.irle_ac(ac_coeffs)
            zz = [dc_coeff] + ac_coeffs
            quantized_block = func.inverse_zigzag(np.array(zz))
            
            block = (func.idct_2d(func.dequantize(quantized_block, Q_Y, quality)) + 128).astype(np.uint8)
            #print(c)
            c+=1
            y[i:i+8, j:j+8] = block
            # if i == 0 and j == 0:
            #     return block
    







    # Дописать Cb, Cr. Разобраться с их размерами 
    # Есть проблема при декодировании AC коэффициентов, некоторые не совпадают. С DC все ок
    
    return y, DC




im_RGB = Image.open("data/Lenna.png")
size = im_RGB.size
im_RGB = np.array(im_RGB)
c, dc= compress(im_RGB)

d, dc2 = decompress(c)
dc, dc2 = np.array(dc), np.array(dc2)
print(dc)
print(dc2)
print(dc[:1024] == dc2)


tmp = Image.new('L', (8, 8))
#c = Image.merge('YCbCr', (Image.fromarray(block), tmp, tmp)).convert('RGB')
fig, ax = plt.subplots(1,2)
ax[0].imshow(Image.fromarray(block))
ax[1].imshow(Image.fromarray(d))
plt.show()
# print(d)
# tmp_chanel = Image.new("L", size, 128)
# y = Image.fromarray(d)
# fig, ax = plt.subplots()
# #im_y = Image.merge("YCbCr", (tmp_chanel, tmp_chanel, Cr)).convert('RGB')
# ax.imshow(y)
# plt.show()
    





    