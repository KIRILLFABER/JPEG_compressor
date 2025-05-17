import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import func  # ваш модуль с функциями JPEG
from config import *  # ваши конфиги
import math

def test_rgb_ycbcr_conversion(im):
    """Проверка корректности преобразования RGB <-> YCbCr"""
    print("\n=== Тест RGB <-> YCbCr ===")
    r, g, b = func.getRGB(im)
    y, cb, cr = func.RGB_to_YCbCr(im)
    
    # Обратное преобразование
    r_back, g_back, b_back = func.YCbCr_to_RGB(y, cb, cr)
    
    # Сравнение с оригиналом
    diff = np.abs(r - r_back) + np.abs(g - g_back) + np.abs(b - b_back)
    print(f"Максимальная разница: {diff.max()} (должна быть < 2)")
    
    # Визуализация
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0,0].imshow(y, cmap='gray'); ax[0,0].set_title('Y')
    ax[0,1].imshow(cb, cmap='gray'); ax[0,1].set_title('Cb')
    ax[0,2].imshow(cr, cmap='gray'); ax[0,2].set_title('Cr')
    ax[1,0].imshow(r_back, cmap='gray'); ax[1,0].set_title('R восстановленный')
    ax[1,1].imshow(g_back, cmap='gray'); ax[1,1].set_title('G восстановленный')
    ax[1,2].imshow(b_back, cmap='gray'); ax[1,2].set_title('B восстановленный')
    plt.show()

def test_dct_idct(block_size=8):
    """Проверка DCT и обратного преобразования"""
    print("\n=== Тест DCT/IDCT ===")
    test_block = np.random.randint(0, 255, (block_size, block_size))
    print("Оригинальный блок:\n", test_block)
    
    # DCT
    dct_block = func.dct_2d(test_block - 128)
    print("\nDCT блок:\n", dct_block.astype(int))
    
    # Обратное DCT
    idct_block = func.idct_2d(dct_block) + 128
    print("\nВосстановленный блок:\n", idct_block.astype(int))
    
    # Проверка точности
    error = np.abs(test_block - idct_block)
    print(f"\nМаксимальная ошибка: {error.max():.2f}")
    print(f"Средняя ошибка: {error.mean():.2f}")

def test_quantization(block_size=8):
    """Проверка квантования"""
    print("\n=== Тест квантования ===")
    test_block = np.random.rand(block_size, block_size) * 100 - 50
    print("Оригинальный блок:\n", test_block.astype(int))
    
    # Квантование
    quantized = func.quantize(test_block, Q_Y, 50)
    print("\nКвантованный блок:\n", quantized.astype(int))
    
    # Обратное квантование
    dequantized = func.dequantize(quantized, Q_Y, 50)
    print("\nРасквантованный блок:\n", dequantized.astype(int))
    
    # Проверка ошибки
    error = np.abs(test_block - dequantized)
    print(f"\nМаксимальная ошибка: {error.max():.2f}")
    print(f"Средняя ошибка: {error.mean():.2f}")

def test_zigzag(block_size=8):
    """Проверка зигзаг-сканирования"""
    print("\n=== Тест зигзаг-сканирования ===")
    test_block = np.arange(block_size*block_size).reshape((block_size, block_size))
    print("Оригинальный блок:\n", test_block)
    
    # Зигзаг
    zz = func.zigzag(test_block)
    print("\nЗигзаг последовательность:\n", zz)
    
    # Обратный зигзаг
    restored = func.inverse_zigzag(zz, block_size, block_size)
    print("\nВосстановленный блок:\n", restored)
    
    # Проверка
    print("\nСовпадает ли восстановленный блок с оригиналом?", 
          np.allclose(test_block, restored))

def test_huffman():
    """Проверка кодирования Хаффмана (упрощенная)"""
    print("\n=== Тест Хаффмана ===")
    test_dc = 10
    prev_dc = 5
    dc_diff = test_dc - prev_dc
    
    # Кодирование DC
    dc_code = func.huffman_encode_dc(test_dc, prev_dc, HA_Y_DC_TABLE)
    print(f"Закодированный DC ({test_dc} - {prev_dc} = {dc_diff}):", dc_code)
    
    # Декодирование DC
    decoded_diff, bits = func.huffman_decode_dc(dc_code[0], HA_Y_DC_TABLE)
    decoded_dc = prev_dc + decoded_diff
    print(f"Декодированный DC: {decoded_dc} (использовано бит: {bits})")
    
    # Проверка
    print("Совпадает?", test_dc == decoded_dc)

def test_full_cycle(im, test_block_size=8):
    """Полный тест на одном блоке"""
    print("\n=== Полный тест на одном блоке ===")
    # Вырезаем тестовый блок
    test_block = im[:test_block_size, :test_block_size, 0]  # берем только красный канал
    
    # Прямое преобразование
    print("\n1. Прямое DCT -> Квантование -> Зигзаг")
    dct = func.dct_2d(test_block - 128)
    quant = func.quantize(dct, Q_Y, 50)
    zz = func.zigzag(quant)
    print("Зигзаг коэффициенты:\n", zz)
    
    # Обратное преобразование
    print("\n2. Обратный зигзаг -> Расквантование -> IDCT")
    restored_quant = func.inverse_zigzag(np.array(zz))
    dequant = func.dequantize(restored_quant, Q_Y, 50)
    idct = func.idct_2d(dequant) + 128
    print("Восстановленный блок:\n", idct.astype(int))
    
    # Ошибка
    error = np.abs(test_block - idct)
    print(f"\nМаксимальная ошибка: {error.max():.2f}")
    print(f"Средняя ошибка: {error.mean():.2f}")
    
    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.subplot(121); plt.imshow(test_block, cmap='gray'); plt.title('Оригинал')
    plt.subplot(122); plt.imshow(idct, cmap='gray'); plt.title('Восстановленный')
    plt.show()

def main():
    # Загрузка тестового изображения
    im = np.array(Image.open('./data/Lenna.png'))
    
    # Запуск всех тестов
    test_rgb_ycbcr_conversion(im)
    test_dct_idct()
    test_quantization()
    #test_zigzag()
    #test_huffman()
    test_full_cycle(im)

if __name__ == "__main__":
    main()