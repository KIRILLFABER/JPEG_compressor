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
    y_blocks = func.split_into_blocks(y)
    Cb_blocks = func.split_into_blocks(Cb)
    Cr_blocks = func.split_into_blocks(Cr)
    #print(np.array(Cb_blocks).shape, '-------')
    comp_data = bytearray()
    h,w,c = im.shape
    size = struct.pack('>HHH', h, w, c)
    comp_data.extend(size)
    #print(size)
    bitstream = ''
    for blocks, Q, channel in [(y_blocks, Q_Y, 'y'), (Cb_blocks, Q_C, 'Cb'), (Cr_blocks, Q_C, 'Cr')]:
        prev_dc = {'y' : 0, 'Cb': 0, 'Cr': 0}
        ha_dc_table = {'y' : HA_Y_DC_TABLE, 'Cb': HA_C_DC_TABLE, 'Cr': HA_C_DC_TABLE}
        ha_ac_table = {'y' : HA_Y_AC_TABLE, 'Cb': HA_C_AC_TABLE, 'Cr': HA_C_AC_TABLE}
        for row in blocks: 
            for block in row:
                dct_matrix = func.dct_2d(block.astype(np.int32) - 128)
                q_matrix = func.quantize(dct_matrix, Q, quality)
                zz = func.zigzag(q_matrix)
                prev = prev_dc[channel]
                DC.append(zz[0] - prev)
                dc_code, prev_dc[channel] = func.huffman_encode_dc(zz[0], prev, ha_dc_table[channel])
                ac_rle = func.rle_ac(zz[1:])
                ac_bits = func.huffman_encode_ac(ac_rle, ha_ac_table[channel])
                AC.append(zz[1:])
                # if blocks == y_blocks:
                #     print(ac, '\t', func.huffman_encode_ac(ac, HA_Y_AC_TABLE))
                full_bits = dc_code + ac_bits
                bitstream += full_bits
                #print(dc_code)
                #print(func.huffman_decode_dc(dc_code, HA_Y_DC_TABLE))

    packed_data, pad_len = func.pack_to_bytes(bitstream)
    comp_data.extend(packed_data)
    comp_data.append(pad_len)
    return comp_data




            
            
        

def decompress(data, quality = 50):
    factor = 4
    DC = []
    AC = []
    orig_h, orig_w, orig_c = struct.unpack('>HHH', data[:6]) 
    data = data[6:]
    #print(orig_h, orig_w, orig_c)
    padded_h = math.ceil(orig_h / 8) * 8
    padded_w = math.ceil(orig_w / 8) * 8
    y = np.zeros((padded_h, padded_w), dtype=np.float32)
    h = math.ceil(math.ceil(orig_h / factor) / 8) * 8
    w = math.ceil(math.ceil(orig_w / factor) / 8) * 8
    Cb = np.zeros((h, w), dtype=np.float32)
    Cr = np.zeros((h, w), dtype=np.float32)
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
            
            ac_coeffs, bits_used = func.huffman_decode_ac(bitstream[current_pos:], HA_Y_AC_TABLE)
            current_pos += bits_used

            ac_coeffs = func.irle_ac(ac_coeffs)
            zz = [dc_coeff] + ac_coeffs
            AC.append(zz[1:])
            quantized_block = func.inverse_zigzag(np.array(zz))
            
            block = func.idct_2d(func.dequantize(quantized_block, Q_Y, quality)) + 128
            #print(c)
            c+=1
            y[i:i+8, j:j+8] = block
            # if i == 0 and j == 0:
            #     return block
    for channel in [Cb, Cr]:
        prev_dc = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                dc_coeff, bits_consumed = func.huffman_decode_dc(bitstream[current_pos:], HA_C_DC_TABLE)
                DC.append(dc_coeff)
                dc_coeff += prev_dc
                prev_dc = dc_coeff
                #print(dc_coeff)
                current_pos += bits_consumed
                
                ac_coeffs, bits_used = func.huffman_decode_ac(bitstream[current_pos:], HA_C_AC_TABLE)
                current_pos += bits_used

                ac_coeffs = func.irle_ac(ac_coeffs)
                zz = [dc_coeff] + ac_coeffs
                AC.append(zz[1:])
                quantized_block = func.inverse_zigzag(np.array(zz))
                
                block = func.idct_2d(func.dequantize(quantized_block, Q_C, quality)) + 128
                #print(block)
                #print(c)
                c+=1
                channel[i:i+8, j:j+8] = block





    Cb = func.upsampling(Cb)
    Cr = func.upsampling(Cr)

    # приведение к оригинальному размеру
    y = y[:orig_h, :orig_w]
    Cb = Cb[:orig_h, :orig_w]
    Cr = Cr[:orig_h, :orig_w]
    print(Cb[Cb > 255])
    # YCbCr to RGB
    r, g, b = func.YCbCr_to_RGB(y, Cb, Cr)
    

                        
    
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)




# im_RGB = Image.open("data/Lenna.png")
# im_RGB = np.array(im_RGB)
# f = open('./comp_data/Lenna', 'wb')
# comp_data = compress(im_RGB, 100)
# f.write(comp_data)
# orig_r, orig_g, orig_b = func.getRGB(im_RGB)
# r, g, b = decompress(comp_data, 100)
# r, g, b = Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)
# decomp_im = Image.merge('RGB', (r, g, b))
# fig, ax = plt.subplots(1, 2)


# ax[0].imshow(decomp_im)
# ax[1].imshow(im_RGB)
# plt.show()
    





    