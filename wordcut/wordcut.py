from wordcut import config
from PIL import Image, ImageDraw
import os

def cut_word_with_size_and_border(output_dir: str, image: Image.Image, font: str, count: int) -> list:
    ret = []
    for row in range(1, config.DOC_DIM[1]-1):
        for col in range(1, config.DOC_DIM[0]-1):
            left = config.BOARDER_SIZE + col*(config.FONT_SIZE+2*config.MARGIN_SIZE)
            upper = config.BOARDER_SIZE + row*(config.FONT_SIZE+2*config.MARGIN_SIZE)
            tmp = image.crop((left,
                              upper,
                              left + config.FONT_SIZE + 2*config.MARGIN_SIZE,
                              upper + config.FONT_SIZE + 2*config.MARGIN_SIZE))
            count += 1
            # print(count)
            tmp.save("{}/{}/{}.png".format(output_dir, font, count))
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

def binaryzation(image: Image.Image, threshold: int):
    Gray = image.convert('L')
    table  =  []
    for i in range(256):
        if  i  <  threshold:
            table.append(0)
        else :
            table.append(1)
    bim = Gray.point(table, '1')
    return bim

if __name__ == "__main__":
    img = Image.open("1.jpg").convert("RGB")
    # img = pretreat(img)
    img = img.resize(config.IMAGE_SIZE)
    output_path = "./output"
    img_list = cut_word_with_size_and_border(output_path, img, config.FONT_NAME_HuaWen, 1)