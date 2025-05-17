import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import struct
from config import *  # Импорт таблиц Хаффмана и матриц квантования
from scipy.fftpack import dct, idct

def getRGB(im):
    """Извлечение каналов RGB из изображения"""
    return im[:,:,0], im[:,:,1], im[:,:,2]

def RGB_to_YCbCr(im):
    """Конвертация RGB в YCbCr с правильными коэффициентами"""
    r, g, b = getRGB(im)
    r, g, b = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    
    return (
        np.clip(y, 0, 255).astype(np.uint8),
        np.clip(cb, 0, 255).astype(np.uint8),
        np.clip(cr, 0, 255).astype(np.uint8)
    )

def YCbCr_to_RGB(y, cb, cr):
    """Конвертация YCbCr в RGB с контролем диапазона"""
    y, cb, cr = y.astype(np.float32), cb.astype(np.float32)-128, cr.astype(np.float32)-128
    
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    
    return (
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8)
    )

def downsampling(channel):
    """Уменьшение разрешения цветовых компонент 4:2:0"""
    return channel[::2, ::2]

def upsampling(channel):
    """Увеличение разрешения цветовых компонент"""
    h, w = channel.shape
    upsampled = np.zeros((h*2, w*2), dtype=channel.dtype)
    upsampled[::2, ::2] = channel
    upsampled[1::2, ::2] = channel
    upsampled[::2, 1::2] = channel
    upsampled[1::2, 1::2] = channel
    return upsampled

def split_into_blocks(channel, block_size=8):
    """Разбиение канала на блоки 8x8"""
    h, w = channel.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    
    padded = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='edge')
    blocks = []
    for i in range(0, padded.shape[0], block_size):
        row = []
        for j in range(0, padded.shape[1], block_size):
            row.append(padded[i:i+block_size, j:j+block_size])
        blocks.append(row)
    return blocks

def dct_2d(block):
    """2D DCT преобразование"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    """Обратное 2D DCT преобразование"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, quantization_table, quality=50):
    """Квантование DCT коэффициентов"""
    if quality <= 0:
        quality = 1
    elif quality > 100:
        quality = 100
    
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    scaled_table = np.floor((quantization_table * scale + 50) / 100)
    scaled_table[scaled_table < 1] = 1
    
    return np.round(block / scaled_table).astype(np.int32)

def dequantize(block, quantization_table, quality=50):
    """Обратное квантование"""
    if quality <= 0:
        quality = 1
    elif quality > 100:
        quality = 100
    
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    scaled_table = np.floor((quantization_table * scale + 50) / 100)
    scaled_table[scaled_table < 1] = 1
    
    return block * scaled_table

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


def inverse_zigzag(zigzag_vector, rows, cols):
    """Обратное зигзаг-сканирование"""
    block = np.zeros((rows, cols))
    index = 0
    for k in range(1-rows, rows):
        diagonal = np.diagonal(block[::-1,:], k)
        diagonal[:] = zigzag_vector[index:index+len(diagonal)]
        index += len(diagonal)
    return block

def rle_ac(ac_coeffs):
    """RLE кодирование AC коэффициентов"""
    rle = []
    zero_count = 0
    for coeff in ac_coeffs:
        if coeff == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                rle.append((15, 0))
                zero_count -= 16
            rle.append((zero_count, coeff))
            zero_count = 0
    rle.append((0, 0))  # EOB
    return rle

def irle_ac(rle_data):
    """Обратное RLE преобразование"""
    ac_coeffs = []
    for run_length, coeff in rle_data:
        if run_length == 0 and coeff == 0:  # EOB
            break
        ac_coeffs.extend([0]*run_length)
        ac_coeffs.append(coeff)
    # Заполняем оставшиеся нули
    ac_coeffs.extend([0]*(63 - len(ac_coeffs)))
    return ac_coeffs

def huffman_encode_ac(rle_pairs, huffman_table):
    bitstream = ""
    for run, value in rle_pairs:
        if (run, value) == (0, 0):  # EOB (End of Block)
            bitstream += huffman_table[(0, 0)]
            break
        
        # Определяем размер значения (size)
        if value == 0:
            size = 0
        else:
            size = max(int(math.log2(abs(value))) + 1, 1)  # Не менее 1 бита
        
        # Получаем код Хаффмана для (run, size)
        try:
            huffman_code = huffman_table[(run, size)]
        except KeyError:
            raise ValueError(f"Не найден код Хаффмана для (run={run}, size={size})")
        
        bitstream += huffman_code
        
        # Кодируем само значение (если не нулевое)
        if size > 0:
            if value > 0:
                bitstream += bin(value)[2:].zfill(size)
            else:
                # Модифицированный two's complement для JPEG
                bitstream += bin((1 << size) + value - 1)[2:].zfill(size)
    
    return bitstream

def huffman_decode_ac(bitstream, huffman_table, max_coeffs=63):
    inv_table = {v: k for k, v in huffman_table.items()}
    rle_pairs = []
    i = 0
    n = len(bitstream)
    
    while i < n and len(rle_pairs) < max_coeffs:
        # Ищем код Хаффмана
        code = ""
        found = None
        for bit in bitstream[i:]:
            code += bit
            if code in inv_table:
                found = inv_table[code]
                break
        
        if found is None:
            raise ValueError(f"Неверный код Хаффмана: {code}")
        
        run, size = found
        i += len(code)
        
        # Обработка EOB
        if (run, size) == (0, 0):
            rle_pairs.append((0, 0))
            break
        
        # Декодирование значения
        value = 0
        if size > 0:
            if i + size > n:
                raise ValueError("Недостаточно битов для значения")
            
            value_bits = bitstream[i:i+size]
            i += size
            
            # Декодирование с учетом знака
            value = int(value_bits, 2)
            if value_bits[0] == '0':  # Отрицательное число
                value = -( (1 << size) - value - 1 )
        
        rle_pairs.append((run, value))
    
    return rle_pairs, i



def huffman_encode_dc(dc, prev_dc, huffman_table):
    dc_diff = dc - prev_dc
    if dc_diff == 0:
        dc_size = 0
    else:
        dc_size = max(int(math.log2(abs(dc_diff))) + 1, 1)  # Не менее 1 бита
    
    dc_code = huffman_table[dc_size]
    
    if dc_size > 0:
        if dc_diff >= 0:
            dc_code += bin(dc_diff)[2:].zfill(dc_size)
        else:
            # Modified two's complement для JPEG
            dc_code += bin((1 << dc_size) + dc_diff - 1)[2:].zfill(dc_size)
    
    return dc_code, dc

def huffman_decode_dc(bitstream, huffman_table):
    inv_table = {v: k for k, v in huffman_table.items()}
    code = ''
    
    for i, bit in enumerate(bitstream):
        code += bit
        if code in inv_table:
            size = inv_table[code]
            if size == 0:
                return 0, i + 1
            
            if i + 1 + size > len(bitstream):
                raise ValueError("Недостаточно битов для DC значения")
            
            value_bits = bitstream[i+1:i+1+size]
            value = int(value_bits, 2)
            
            # Modified two's complement декодирование
            if value_bits[0] == '1':  # Положительное
                pass
            else:  # Отрицательное
                value = -((1 << size) - value - 1)
            
            return value, i + 1 + size
    
    raise ValueError("Неверный код Хаффмана для DC")


def pack_to_bytes(bitstream):
    """Упаковка битовой строки в байты"""
    pad_len = (8 - len(bitstream) % 8) % 8
    bitstream += '0' * pad_len
    byte_array = bytearray()
    for i in range(0, len(bitstream), 8):
        byte = bitstream[i:i+8]
        byte_array.append(int(byte, 2))
    return byte_array, pad_len

def compress(im, quality=50):
    """Основная функция сжатия JPEG"""
    y, cb, cr = RGB_to_YCbCr(im)
    cb = downsampling(cb)
    cr = downsampling(cr)
    
    y_blocks = split_into_blocks(y)
    cb_blocks = split_into_blocks(cb)
    cr_blocks = split_into_blocks(cr)
    
    comp_data = bytearray()
    h, w, c = im.shape
    size = struct.pack('>HHH', h, w, c)
    comp_data.extend(size)
    
    bitstream = ''
    for blocks, Q, channel in [(y_blocks, Q_Y, 'y'), 
                              (cb_blocks, Q_C, 'cb'), 
                              (cr_blocks, Q_C, 'cr')]:
        prev_dc = 0
        for row in blocks:
            for block in row:
                dct_matrix = dct_2d(block - 128)
                q_matrix = quantize(dct_matrix, Q, quality)
                zz = zigzag(q_matrix)
                
                dc_diff = zz[0] - prev_dc
                dc_code, prev_dc = huffman_encode_dc(zz[0], prev_dc, 
                                                   HA_Y_DC_TABLE if channel == 'y' else HA_C_DC_TABLE)
                ac_rle = rle_ac(zz[1:])
                ac_bits = huffman_encode_ac(ac_rle, 
                                          HA_Y_AC_TABLE if channel == 'y' else HA_C_AC_TABLE)
                
                bitstream += dc_code + ac_bits
    
    packed_data, pad_len = pack_to_bytes(bitstream)
    comp_data.extend(packed_data)
    comp_data.append(pad_len)
    return comp_data

def decompress(data, quality=50):
    """Основная функция распаковки JPEG"""
    orig_h, orig_w, orig_c = struct.unpack('>HHH', data[:6])
    data = data[6:]
    
    # Инициализация массивов
    padded_h = math.ceil(orig_h / 8) * 8
    padded_w = math.ceil(orig_w / 8) * 8
    y = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # Для хроматических каналов
    factor = 2
    h = math.ceil(math.ceil(orig_h / factor) / 8) * 8
    w = math.ceil(math.ceil(orig_w / factor) / 8) * 8
    cb = np.zeros((h, w), dtype=np.float32)
    cr = np.zeros((h, w), dtype=np.float32)
    
    pad_len = data[-1]
    data = data[:-1]
    bitstream = ''.join(f'{byte:08b}' for byte in data)
    if pad_len != 0:
        bitstream = bitstream[:-pad_len]
    
    current_pos = 0
    
    # Декодирование Y канала
    prev_dc = 0
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            dc_diff, bits = huffman_decode_dc(bitstream[current_pos:], HA_Y_DC_TABLE)
            current_pos += bits
            dc_coeff = prev_dc + dc_diff
            prev_dc = dc_coeff
            
            ac_coeffs, bits = huffman_decode_ac(bitstream[current_pos:], HA_Y_AC_TABLE)
            current_pos += bits
            ac_coeffs = irle_ac(ac_coeffs)
            
            zz = [dc_coeff] + ac_coeffs
            quant_block = inverse_zigzag(zz, 8, 8)
            dequant_block = dequantize(quant_block, Q_Y, quality)
            block = idct_2d(dequant_block) + 128
            y[i:i+8, j:j+8] = np.clip(block, 0, 255)
    
    # Декодирование Cb и Cr каналов
    for channel, table in [(cb, HA_C_DC_TABLE), (cr, HA_C_DC_TABLE)]:
        prev_dc = 0
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                dc_diff, bits = huffman_decode_dc(bitstream[current_pos:], table)
                current_pos += bits
                dc_coeff = prev_dc + dc_diff
                prev_dc = dc_coeff
                
                ac_coeffs, bits = huffman_decode_ac(bitstream[current_pos:], HA_C_AC_TABLE)
                current_pos += bits
                ac_coeffs = irle_ac(ac_coeffs)
                
                zz = [dc_coeff] + ac_coeffs
                quant_block = inverse_zigzag(zz, 8, 8)
                dequant_block = dequantize(quant_block, Q_C, quality)
                block = idct_2d(dequant_block) + 128
                channel[i:i+8, j:j+8] = np.clip(block, 0, 255)
    
    # Upsampling и обрезка
    cb = upsampling(cb)[:orig_h, :orig_w]
    cr = upsampling(cr)[:orig_h, :orig_w]
    y = y[:orig_h, :orig_w]
    
    # Конвертация в RGB
    r, g, b = YCbCr_to_RGB(y, cb, cr)
    return np.dstack((r, g, b)).astype(np.uint8)

# Пример использования
if __name__ == "__main__":
    # Загрузка изображения
    im = np.array(Image.open('data/Lenna.png'))
    
    # Сжатие
    compressed = compress(im, quality=75)
    
    # Распаковка
    decompressed = decompress(compressed, quality=75)
    
    
    # Визуализация
    plt.figure(figsize=(12,6))
    plt.subplot(121); plt.imshow(im); plt.title('Original')
    plt.subplot(122); plt.imshow(decompressed); plt.title('Decompressed')
    plt.show()