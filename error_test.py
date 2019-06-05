import cut
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('test_dir', action="store", help='test dir that contains pretreat test pictures')
parser.add_argument('cut_result_dir', action="store", help='cut result dir that used to contained cutted test pictures')


code1 = '100001000111101000110000100111100010111011001000001011000111010100101010110111110100010100110101011001100101100111100101011011100110000110001000'
code2 = '111110001001011001011101100001001101100101100010100011011100100010011000011100010010011011110110110000000010100010001010000000000110100100111111'
code3 = '110101111000111100010000110111110111100111010111100100010001011100010000010111101111001001100101000100101001001111110111111110001011110111110101'


test1 = "test1"
test2 = "test2"
test3 = "test3"

code = [code1, code2, code3]
test = [test1, test2, test3]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

def do_predict(word):
    outputs = net(word)
    _, predicted = torch.max(outputs, 1)
    return predicted


def decode(word_cnt, decode_path):
    decode_str = ''
    with torch.no_grad():
        for cnt in range(1, word_cnt+1):
            path = decode_path + '/' +str(cnt)+'.png'
            # print(path)
            word = trans(Image.open(path))
            word = torch.stack((word, word, word), 0).to(device)
            predict = do_predict(word)
            decode_str += str(np.array(predict)[0])    
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
def cut_images(test_dir, cut_result_dir):
    for i in range(1):
        input_dir = "{}/{}".format(test_dir, test[i])
        output_dir = "{}/{}".format(cut_result_dir, test[i])
        os.mkdir(output_dir)
        cut.cutInputImages(input_dir, cut_result_dir, test[i])
    # input_dir = "./input"
    # output_dir = "./output"
    # os.system('rm -rf ' + output_dir)
    # os.mkdir(output_dir)
    # cut.cutInputImages(input_dir, output_dir)

def main(test_dir, cut_result_dir):
    cut_images(test_dir, cut_result_dir)
    for i in range(1):
        image_folder = "{}/{}".format(cut_result_dir, test[i])
        pic_list = os.listdir(image_folder)
        for pic in pic_list:
            decode_path = "{}/{}".format(image_folder, pic)
            font_num = len(os.listdir(decode_path))
            rslt = decode(font_num, decode_path)
            cal_accu(rslt, code[i])
    # one_test
    # output_dir = "../output"
    # decode_path = "{}/{}".format(output_dir, 0)
    # font_num = len(os.listdir(decode_path))
    # rslt = decode(font_num, decode_path)
    # cal_accu(rslt, "0"*140)            

if __name__ == "__main__":
    # args = parser.parse_args()
    # cut_result_dir = args.cut_result_dir
    # test_dir = args.test_dir
    # if not os.path.exists(cut_result_dir):
    #     os.mkdir(cut_result_dir)
    net = torch.load('../Model/last.pkl', map_location='cpu' )
    # net = net.to(device)
    net.eval()
    cut.cutInputImage('../cut_rslt/1.jpg', '../test', '0')
    # main(args, cut_result_dir)
    decode_path = "../test/0"
    font_num = len(os.listdir(decode_path))
    rslt = decode(font_num, decode_path)
    print(rslt)
