from flask import Flask, request
from pretreat import crop_image
from PIL import Image
from wordcut import cut
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from torchvision import transforms, models, datasets
from coder.Coder import MyRS

app = Flask(__name__)
# input dir store images as username.png
input_dir = "./input"
# ouput dir contains dirs as username
output_dir = "./output"


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
            print(path)
            word = trans(Image.open(path))
            word = torch.stack((word, word, word), 0).to(device)
            predict = do_predict(word)
            decode_str += str(np.array(predict)[0])    
    return decode_str


# mk inputdir and outputdir/username
def mk_user_dir(username, input_dir = input_dir, output_dir = output_dir):
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    user_output_dir = '{}/{}'.format(output_dir, username)

    if os.path.exists(user_output_dir):
        os.system('rm -rf ' + user_output_dir)

    os.mkdir(user_output_dir)


@app.route("/", methods = ['POST'])
def predict():
    # get username and file from request
    input_img = request.files['image']
    username = request.form.get('user')

    # crop input image and make output dir 
    mk_user_dir(username)
    input_img = Image.open(input_img)
    pretreat_img = crop_image(input_img)
    input_img_name = '{}/{}.png'.format(input_dir, username)
    pretreat_img.convert('RGB').save(input_img_name, quality=95)

    # cut input image and save in ouput dir
    cut_rslt = cut.cutInputImages(input_img_name, output_dir, username)
    decode_path = '{}/{}'.format(output_dir, username)
    decode_str = decode(cut_rslt, decode_path)
    print(decode_str)
    decode_list = [int(i) for i in list(decode_str)]
    # temp len as 128
    decode_rslt = coder.decode(decode_list[:128])
    print(decode_rslt)
    return decode_rslt



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        net = torch.load('../Model/ResnetModel_last.pkl')
    else:
        net  = torch.load('../Model/ResnetModel_last.pkl',  map_location='cpu')
    net = net.to(device)
    net.eval()
    coder = MyRS()
    app.run(host='0.0.0.0', port=8080)