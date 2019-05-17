import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets
from wordcut.wordcut import cutInputImages


# TODO: transform from PIL.IMAGE to Tensor with normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

from flask import Flask,request,render_template,render_template_string, url_for, redirect
import json

app = Flask(__name__)


def split_words(input_path, output_path):
    return cutInputImages(input_path, output_path)

def do_predict(word):
    outputs = net(word)
    _, predicted = torch.max(outputs, 1)
    return predicted


def show_upload_form():
    # return render_template('test.html', name='yjw')
    # return render_template_string('hhh')
    return render_template('predict.html')

@app.route('/')
def welcome():
    return render_template_string('Welcome to Font-code alpha!')

@app.route('/decode')
@app.route('/decode/<decode_str>/<su1>/<su2>')
def decode(decode_str=None):
    return render_template('decode.html', decode_str = decode_str)

@app.route('/predict/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        image.save('./input/'+image.filename)
        input_path = './input/'
        output_path = './output/'
        
        word_cnt = split_words(input_path, output_path)
        # predict images which is separated in folders
        print('word_cnt', word_cnt)
        
        decode_str = ''
        with torch.no_grad():
            for cnt in range(1, word_cnt+1):
                word = trans(Image.open(output_path+'0/'+str(cnt)+'.png').convert('RGB'))
                word = torch.stack((word, word, word), 0)
                predict = do_predict(word)
                decode_str += str(np.array(predict)[0])


        # inputs_images = datasets.ImageFolder(root=output_path, transform = trans)
        # print(inputs_images.imgs)
        # decode_str = ''
        # with torch.no_grad():
        #     for data in inputs_images:
        #         images, labels = data
        #         images = torch.stack((images, images, images), 0)
        #         predicted = do_predict(images)
        #         decode_str += str(np.array(predicted)[0])
                # c = (predicted == labels).squeeze()
                # if c.dim() < 1:
                #     break
                # for i in range(3):
                #     label = labels[i]
                #     class_correct[label] += c[i].item()
                #     class_total[label] += 1
        print(decode_str)
        real_1 = '11101001101100011001101111011010110010011100101110111110101000000011000100001010110000000101001001010111001110010001111010010101010101110001'
        real_2 = '11010100001111001110100000000001110101101001111010100001110000000010101000100100111101100110111100000011000101001111000100011100101100000010'
        cnt = 0
        for i in range(len(real_1)):
            if real_1[i] == decode_str[i]:
                cnt += 1
        bit_success_rate_1 = cnt / len(real_1)
        cnt = 0
        for i in range(len(real_2)):
            if real_2[i] == decode_str[i]:
                cnt += 1
        bit_success_rate_2 = cnt / len(real_2)
        print('bit_success_rate_1:', bit_success_rate_1)
        print('bit_success_rate_2:', bit_success_rate_2)

        return json.dumps({
            'decode': decode_str,
            'success1': bit_success_rate_1,
            'success2': bit_success_rate_2
        })

        # predict single word
        
        # word = trans(Image.open('./images/'+image.filename).convert('RGB'))
        # word = torch.stack((word, word, word), 0)
        # predict = do_predict(word)
        # decode_str = str(np.array(predict)[0])
    else:
        return show_upload_form()

if __name__ == '__main__':

    # restore entire net
    net = torch.load('net1.0.pkl', map_location='cpu')
    net.eval()
    app.run(host='0.0.0.0', debug=True)

   




