import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from PIL import Image 
from torchvision import transforms, models, datasets


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


def split_words():
    pass

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
@app.route('/decode/<decode_str>')
def decode(decode_str=None):
    return render_template('decode.html', decode_str = decode_str)

@app.route('/predict/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        image.save('./images/'+image.filename)

        # predict images which is separated in folders

        inputs_images = datasets.ImageFolder(root='./images', transform = trans)
        decode_str = ''
        with torch.no_grad():
            for data in inputs_images:
                images, labels = data
                images = torch.stack((images, images, images), 0)
                predicted = do_predict(images)
                decode_str += str(np.array(predicted)[0])
                # c = (predicted == labels).squeeze()
                # if c.dim() < 1:
                #     break
                # for i in range(3):
                #     label = labels[i]
                #     class_correct[label] += c[i].item()
                #     class_total[label] += 1
        print(decode_str)
        return json.dumps({
            'decode': decode_str
        })

        # predict single word
        
        # word = trans(Image.open('./images/'+image.filename).convert('RGB'))
        # word = torch.stack((word, word, word), 0)
        # predict = do_predict(word)
        # decode_str = str(np.array(predict)[0])
    else:
        return show_upload_form()

if __name__ == '__main__':
    #restore entire net
    net = torch.load('net1.0.pkl', map_location=lambda storage, loc: storage)
    net.eval()
    app.run(host='0.0.0.0', debug=True)

   




