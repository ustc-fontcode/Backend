import cut
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets

code1 = '010100100111100010010110011001001101011010111100001111111010100110000100111110001101101101001000001001010011010001010011010110100001011100010001011001101111001010111000100101000100011011011000'
code2 = '110011010010110011111111100110100101110111100110000101101010000000001010111011000011011101010101010111110110011101111101111001000101001000110101111011000111000001100100000100010100001001110110'
code3 = '011011000000111111111000111001111100010010001111100110111011101010011000011001111100010111101111011110010001101010100000001101110000001001101001001110000101011010001001111011000010001010011100'

test_dir = "../error_test/"
test1 = "test1"
test2 = "test2"
test3 = "test3"

code = [code1, code2, code3]
test = [test1, test2, test3]

cut_result_dir = "../error_results"
os.system('rm -rf ' + cut_result_dir)
os.mkdir(cut_result_dir)


def do_predict(word):
    outputs = net(word)
    _, predicted = torch.max(outputs, 1)
    return predicted


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

def decode(word_cnt, decode_path):
    decode_str = ''
    with torch.no_grad():
        for cnt in range(1, word_cnt+1):
            word = trans(Image.open(decode_path + '/' +str(cnt)+'.png').convert('RGB'))
            word = torch.stack((word, word, word), 0)
            predict = do_predict(word)
            decode_str += str(np.array(predict)[0])    
    print(decode_str)
    return decode_str

def cal_accu(decode_str, answer_str):
    cnt = 0
    for i in range(len(decode_str)):
        if answer_str[i] == decode_str[i]:
            cnt += 1
    bit_success_rate = cnt / len(decode_str)
    print(bit_success_rate)
    return bit_success_rate

# cut imgages
def cut_images():
    for i in range(3):
        input_dir = test_dir + test[i]
        output_dir = "{}/{}".format(cut_result_dir, test[i])
        os.mkdir(output_dir)
        cut.cutInputImages(input_dir, output_dir)

def main():
    cut_images()
    for i in range(3):
        image_folder = "{}/{}".format(cut_result_dir, test[i])
        pic_list = os.listdir(image_folder)
        for pic in pic_list:
            decode_path = "{}/{}".format(image_folder, pic)
            font_num = len(os.listdir(decode_path))
            rslt = decode(font_num, decode_path)
            cal_accu(rslt, code[i])

if __name__ == "__main__":
    net = torch.load('net1.0.pkl', map_location='cpu')
    net.eval()
    main()
