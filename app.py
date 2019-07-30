import shutil

import flask
from flask import Flask, request
from pretreat import crop_image
from PIL import Image
from pathlib import Path
# from wordcut import cut
import cut
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from torchvision import transforms, models, datasets
from coder.Coder import MyVote
from coder.generator import generate_doc_with_code_and_bias

app = Flask(__name__)
# input dir store images as username.png
input_dir = "./input"
# ouput dir contains dirs as username
output_dir = "./output"
# to generate the doc
gen_dir = "./gen"

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
        for cnt in range(1, word_cnt + 1):
            path = decode_path + '/' + str(cnt) + '.png'
            print(path)
            word = trans(Image.open(path))
            word = torch.stack((word, word, word), 0).to(device)
            predict = do_predict(word)
            decode_str += str(np.array(predict)[0])
    return decode_str


# mk inputdir and outputdir/username
def mk_user_dir(username, input_dir=input_dir, output_dir=output_dir, gen_dir=gen_dir):
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    user_output_dir = '{}/{}'.format(output_dir, username)

    if os.path.exists(user_output_dir):
        # os.system('rm -rf ' + user_output_dir)
        shutil.rmtree(user_output_dir)

    os.mkdir(user_output_dir)


@app.route("/", methods=["GET"])
def test():
    return "hello world"


@app.route("/generate/", methods=["POST"])
def generate() -> "List (Image.Image)":
    """
    采用Post 的方式
    user：
    text： 是文档中的文字， 应该保证大于144个字
    bits：编码进的消息， 4bits默认
    返回静态的二进制文件： png格式 如何接受可以看testApp代码

    """
    # get username and the str info
    username = request.form.get("user")
    text = request.form.get("text")
    bits = request.form.get("bits")
    assert len(bits) == 4
    bits = [int(x) for x in bits]
    coder = MyVote(c=len(bits))
    bits = coder.encode(bits)
    mk_user_dir(username, gen_dir=gen_dir)
    doc = generate_doc_with_code_and_bias(bits=bits,
                                          text=text,
                                          font_names=['coder/data/fonts/HuaWenSun.ttf',
                                                      'coder/data/fonts/MicroSun.ttf'],
                                          bias={0: (0, -8)})
    gen_img_name = Path(gen_dir) / (username + ".png")
    doc[0].save(gen_img_name)
    print(str(gen_img_name))
    return flask.send_file(str(gen_img_name), mimetype='image/png')



@app.route("/predict/", methods=['POST'])
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
        net = torch.load('../Model/ResnetModel_last.pkl', map_location='cpu')
    net = net.to(device)
    net.eval()
    coder = MyVote()
    app.run(host='0.0.0.0', port=8080)
