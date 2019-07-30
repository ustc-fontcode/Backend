import PIL
import cv2
from pathlib import Path
import config
from PIL import Image, ImageOps
import os
import numpy as np
from pathlib import Path
import pretreat


def cut_word_with_size_and_border(output_dir: str,
                                  image: Image.Image,
                                  sub_dir: str,
                                  count: int) -> list:
    ret = []
    for row in range(1, config.DOC_DIM[1] - 1):
        for col in range(1, config.DOC_DIM[0] - 1):
            left = config.BOARDER_SIZE \
                   + col * (config.FONT_SIZE + 2 * config.MARGIN_SIZE)
            upper = config.BOARDER_SIZE \
                    + row * (config.FONT_SIZE + 2 * config.MARGIN_SIZE)
            tmp = image.crop((left,
                              upper,
                              left + config.FONT_SIZE \
                              + 2 * config.MARGIN_SIZE,
                              upper + config.FONT_SIZE \
                              + 2 * config.MARGIN_SIZE))
            count += 1
            # print(count)
            tmp.save("{}/{}/{}.png".format(output_dir, sub_dir, count))
            ret.append(tmp)
    return ret


def cutInputImages(input_path, output_path):
    # clean output dir when start cutting
    os.system('rm -rf ' + output_path + '*')
    input_images = os.listdir(input_path)
    input_cnt = 0

    for image_name in input_images:
        img = Image.open(input_path + image_name).convert("RGB")
        img = img.resize(config.IMAGE_SIZE)
        output_folder = str(input_cnt)
        os.makedirs(output_path + output_folder)
        img_list = cut_word_with_size_and_border(output_path, img, output_folder, 0)
        input_cnt += 1
    return len(img_list)


def cut_word_via_statics(output_dir: str,
                         img: Image.Image,
                         sub_dir: str,
                         count: int):
    """
    参数和cut_word_with_size_and_border() 一样
    :param output_dir:
    :param img:
    :param sub_dir:
    :param count:
    :return:
    """
    img_origin = img

    # 二值化
    img_grey = img.convert('L')
    img_grey = ImageOps.invert(img_grey)
    img_binary = binaryzation(img_grey, 180)

    # 开操作去噪声
    img_binary = img_binary.convert("RGB")
    img_binary = cv2.cvtColor(np.array(img_binary), cv2.COLOR_RGB2GRAY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(img_binary, kernel1)
    dilated = cv2.dilate(eroded, kernel2)

    img_binary = binaryzation(Image.fromarray(dilated), 180)
    img_binary_array = np.asarray(img_binary)
    ret_w = []
    ret_h = []
    w_sum = np.sum(img_binary_array, axis=0)
    w_sum = [x - 10 if (x - 10) >= 0 else 0 for x in w_sum]
    h_sum = np.sum(img_binary_array, axis=1)
    h_sum = [x - 10 if (x - 10) >= 0 else 0 for x in h_sum]
    start = 4
    isword = True
    for i, w in enumerate(w_sum):
        if i < 15:
            continue
        if i > len(w_sum) - 15:
            continue
        if isword and w > 8:
            start = i
            isword = False
        elif i - start > 15 and (not isword) and w < 8:
            end = i
            isword = True
            ret_w.append((start, end))
    start = 4
    isword = True
    for i, h in enumerate(h_sum):
        if i < 15:
            continue
        if i > len(h_sum) - 15:
            continue
        if isword and h > 10:
            start = i
            isword = False
        elif i - start > 20 and (not isword) and h < 2:
            end = i
            isword = True
            ret_h.append((start, end))

    # 对字符进行切分
    result = []
    output_path = Path(output_dir)
    for w_block in ret_w:
        for h_block in ret_h:
            if abs((w_block[1] - w_block[0]) - (h_block[1] - h_block[0])) \
                    > 0.5 * (w_block[1] - w_block[0]) and \
                    abs((w_block[1] - w_block[0]) - (h_block[1] - h_block[0])) \
                    > 0.5 * (h_block[1] - h_block[0]):
                continue
            tmp = img_origin.crop((w_block[0],
                                   h_block[0],
                                   w_block[1],
                                   h_block[1]))
            count += 1
            tmp.save(output_path /
                     "{}".format(sub_dir) /
                     "{}.png".format(count))
            result.append(tmp)
    return result


def binaryzation(image: Image.Image, threshold: int):
    Gray = image.convert('L')
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    bim = Gray.point(table, '1')
    return bim
    # img_gray = cv2.colorChange(np.asarray(image), cv2.COLOR_RGB2GRAY)
    # img_gray = np.asarray(image)
    # def OTSU(img_gray):
    #     max_g = 0
    #     suitable_th = 0
    #     th_begin = 0
    #     th_end = 256
    #     for threshold in range(th_begin, th_end):
    #         bin_img = img_gray > threshold
    #         bin_img_inv = img_gray <= threshold
    #         fore_pix = np.sum(bin_img)
    #         back_pix = np.sum(bin_img_inv)
    #         if 0 == fore_pix:
    #             break
    #         if 0 == back_pix:
    #             continue
    #         w0 = float(fore_pix) / img_gray.size
    #         u0 = float(np.sum(img_gray * bin_img)) / fore_pix
    #         w1 = float(back_pix) / img_gray.size
    #         u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
    #         # intra-class variance
    #         g = w0 * w1 * (u0 - u1) * (u0 - u1)
    #         if g > max_g:
    #             max_g = g
    #             suitable_th = threshold
    #     return suitable_th
    # img_ada_gaussian = cv2.adaptiveThreshold(np.array(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #                                          145,
    #                                          3)
    # return Image.fromarray(img_ada_gaussian)


if __name__ == "__main__":
    input_path = Path("coder") / "data" / "train_data"
    output_path = Path("coder") / "data" / "train_data_cut"

    for font_name in os.listdir(input_path):
        for img_name in os.listdir(input_path / font_name):
            img = Image.open(input_path / font_name / img_name).convert("RGB")
            img = pretreat.crop_image(img)
            try:
                os.mkdir(output_path / font_name / img_name)
            except FileExistsError:
                print("exist")
            img_list = cut_word_via_statics(output_path / font_name, img, img_name, 0)
            print(img_name)

    '''
    img = Image.open("test_data\\test.jpg").convert("RGB")
    output_dir = "test_data\\output\\"
    sub_dir = config.FONT_NAME_HuaWen
    cut_word_via_statics(output_dir, img, sub_dir, 0)
    '''
    # img = pretreat(img)
    """
    img = img.resize(config.IMAGE_SIZE)
    output_path = "./output"
    img_list = cut_word_with_size_and_border(output_path, img, config.FONT_NAME_HuaWen, 1)
    """
