import cut
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets

code1 = '01001011000001010100011000110000000000100101011101101000101111110011001110000001011100011110100101010111010111011010011010111101011101011000'
code2 = '11110101111110100110100011100010110110101000010100110000011001100111101010100111101010000111101111101001111011101001111001111111000110010010'
code3 = '011011000000111111111000111001111100010010001111100110111011101010011000011001111100010111101111011110010001101010100000001101110000001001101001001110000101011010001001111011000010001010011100'

code5 = '000101101101111101100110000010100011101010011110011000011100110011101000001111000111000010111001111010011010000100011101011011000000101010010111'

test_dir = "./datasets/new_low/new_low/"
test1 = "test1"
test2 = "test2"
test3 = "test3"

code = [code1, code2, code3]
# test = [test1, test2, test3]
test = [test1]

cut_result_dir = "./datasets/error_results"
os.system('rm -rf ' + cut_result_dir)
os.mkdir(cut_result_dir)


def do_predict(word):
    outputs = net(word)
    _, predicted = torch.max(outputs, 1)
    return predicted


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

def decode(word_cnt, decode_path):
    decode_str = ''
    with torch.no_grad():
        for cnt in range(1, word_cnt+1):
            path = decode_path + '/' +str(cnt)+'.png'
            print(path)
            word = trans(Image.open(path))
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
    for i in range(1):
        input_dir = test_dir + test[i]
        output_dir = "{}/{}".format(cut_result_dir, test[i])
        os.mkdir(output_dir)
        cut.cutInputImages(input_dir, output_dir)
    # input_dir = "./input"
    # output_dir = "./output"
    # os.system('rm -rf ' + output_dir)
    # os.mkdir(output_dir)
    # cut.cutInputImages(input_dir, output_dir)

def main():
    cut_images()
    for i in range(1):
        image_folder = "{}/{}".format(cut_result_dir, test[i])
        pic_list = os.listdir(image_folder)
        for pic in pic_list:
            decode_path = "{}/{}".format(image_folder, pic)
            font_num = len(os.listdir(decode_path))
            rslt = decode(font_num, decode_path)
            cal_accu(rslt, code5)
    # one_test
    # output_dir = "../output"
    # decode_path = "{}/{}".format(output_dir, 0)
    # font_num = len(os.listdir(decode_path))
    # rslt = decode(font_num, decode_path)
    # cal_accu(rslt, "0"*140)            

if __name__ == "__main__":
    net = torch.load('./ResnetModel_last.pkl', map_location='cpu')
    net.eval()
    main()
