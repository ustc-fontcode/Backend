from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import numpy as np
import cv2
import argparse
from coder import read_chinese
from coder import config

parser = argparse.ArgumentParser()

parser.add_argument('--resolution', action="store", help='generate a specific resolution datasets')
parser.add_argument('--mode', action="store", help='generate mode: train or error-test')

def generate_doc_with_code_and_bias(bits: list,
                                    text: str,
                                    font_names: list,
                                    bias={}) -> list:
    """Generate document with given text, bits to be encoded, and font names.
    When text is longer than bits, font_names[0] will be used.
    When text is shorter than bits, extra bits will be ignored.
    Params:
        bits: List of integers, e.g. [0, 1, 0, 0]. Don't have to be 0 or 1, because we can have more than two kinds of fonts.
        text: e.g. "你好世界".
        font_names: Name of font, or path of font file.
        bias:
    Returns:
        List of PIL.Image.
    """

    fonts = [ImageFont.truetype(f, config.FONT_SIZE) for f in font_names]
    ret = []
    row = col = 0
    draw = None
    tbias = [(0, 0) for _ in range(len(font_names))]
    for k, v in bias.items():
        tbias[k] = v
    for cursor in range(len(text)):
        # New page
        if row == 0 and col == 0:
            img = Image.new("RGB", config.DOC_IMAGE_SIZE, config.FOREGROUND_COLOR)
            draw = ImageDraw.Draw(img)
            draw.rectangle(xy=[
                (config.BLANK_SIZE[0], config.BLANK_SIZE[1]),
                (config.DOC_IMAGE_SIZE[0] - config.BLANK_SIZE[0],
                 config.DOC_IMAGE_SIZE[1] - config.BLANK_SIZE[1])
            ],
                           outline=config.BACKGROUND_COLOR,
                           width=config.BOARDER_SIZE)
            ret.append(img)
        if cursor >= len(bits):
            bit = 0
        else:
            bit = bits[cursor]
        draw.text((config.BLANK_SIZE[0] + config.BOARDER_SIZE + (col + 1) *
                   (config.FONT_SIZE + 2 * config.MARGIN_SIZE) +
                   config.MARGIN_SIZE + tbias[bit][0], config.BLANK_SIZE[1] + config.BOARDER_SIZE +
                   (row + 1) * (config.FONT_SIZE + 2 * config.MARGIN_SIZE) +
                   config.MARGIN_SIZE + tbias[bit][1]),
                  text[cursor],
                  config.BACKGROUND_COLOR,
                  font=fonts[bit])
        tmp = col + 1
        col = tmp % (config.DOC_DIM[0] - 2)
        row = (row + tmp // (config.DOC_DIM[0] - 2)) % (config.DOC_DIM[1] - 2)
    return ret


def generate_doc(chars: str, font_name: str):
    print(font_name)
    img = Image.new("RGB", config.DOC_IMAGE_SIZE, config.FOREGROUND_COLOR)
    font = ImageFont.truetype(font_name, config.FONT_SIZE)
    draw = ImageDraw.Draw(img)
    draw.rectangle(xy=[(config.BLANK_SIZE[0], config.BLANK_SIZE[1]),
                       (config.DOC_IMAGE_SIZE[0] - config.BLANK_SIZE[0],
                        config.DOC_IMAGE_SIZE[1] - config.BLANK_SIZE[1])],
                   outline=config.BACKGROUND_COLOR,
                   width=config.BOARDER_SIZE)
    try:
        cursor = 0
        for row in range(1, config.DOC_DIM[1] - 1):
            for col in range(1, config.DOC_DIM[0] - 1):
                draw.text((config.BLANK_SIZE[0] + config.BOARDER_SIZE + col *
                           (config.FONT_SIZE + 2 * config.MARGIN_SIZE) +
                           config.MARGIN_SIZE,
                           config.BLANK_SIZE[1] + config.BOARDER_SIZE + row *
                           (config.FONT_SIZE + 2 * config.MARGIN_SIZE) +
                           config.MARGIN_SIZE),
                          chars[cursor],
                          config.BACKGROUND_COLOR,
                          font=font)
                cursor += 1
    except IndexError:
        pass
    return img


def generate_iter(chars: str, font_name: str):
    font_name = "data/fonts/{}.ttf".format(font_name)
    for i in range(0, len(chars), config.DOC_DIM[0] * config.DOC_DIM[1]):
        image = generate_doc(chars[i:], font_name)
        #     display(generate_doc(chars[i:], "fonts/{}".format(FONT_NAME), FONT_SIZE))
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
        cv2.waitKey()


def generate_doc_with_random_code(n: int):
    bits = np.random.randint(2, size=(n, ))
    print(bits)
    text = read_chinese.read_chinese3000()
    for i in range(len(text) // n):
        text_epo = text[i * n: i * n + n]
        # docs = generate_doc_with_code_and_bias(bits,
        #                                        text_epo, ['data/fonts/HuaWenSun.ttf', 'data/fonts/MicroSun.ttf'],
        #                                        bias={1: (0, 0)})
        
        docs = generate_doc_with_code_and_bias(bits,
                                               text_epo, ['data/fonts/HuaWenSun.ttf', 'data/fonts/MicroSun.ttf'],
                                               bias={0: (0, -8)})
        
        for img in docs:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img)
            cv2.waitKey()


if __name__ == "__main__":
    
    sep_img = Image.open("data/seperate.jpeg")
    # 生成文档，敲回车生成下一页
    
    chars = read_chinese.read_chinese3000()
    chars = chars * 10
    # generate_doc(chars, config.FONT_NAME_Fangzheng)
    NUM = len(chars)
    
    code = [0 for _ in range(NUM)]
    text = chars[:NUM]

    args = parser.parse_args()

    if args.mode == 'train':
        # 微软仿宋
        doc = generate_doc_with_code_and_bias(code,
                                            text, ['data/fonts/MicroSun.ttf'],
                                            bias={0: (0, 0)})
        for img in doc:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img)
            cv2.waitKey()

        sep_img = cv2.cvtColor(np.asarray(sep_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("sep", sep_img)
        cv2.waitKey()

        # 华文仿宋
        doc = generate_doc_with_code_and_bias(code,
                                            text, ['data/fonts/HuaWenSun.ttf'],
                                            bias={0: (0, -8)})
        for img in doc:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imshow("img", img)
            cv2.waitKey()
    
    elif args.mode == 'error':   
        generate_doc_with_random_code((config.DOC_DIM[0] - 2) * (config.DOC_DIM[1] - 2))
    