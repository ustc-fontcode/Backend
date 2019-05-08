from wordcut import config
import pytesseract
from PIL import Image, ImageDraw
from wordcut.Character import ChineseCharacter
import wordcut.pretreat
import os

def cut_word_with_tesseract(img: Image.Image, font: str) -> list:
    # img = img.resize((1000, 1000))
    draw = ImageDraw.Draw(img)
    print(img.size)
    boxes = pytesseract.image_to_boxes(img, lang="chi_sim").split("\n")
    print(pytesseract.image_to_string(img, lang="chi_sim"))
    chars = []
    for b in boxes:
        if b == "":
            continue
        char = ChineseCharacter(b, img, font)
        draw.rectangle([(char.x1, char.y1), (char.x2, char.y2)], outline="red")
    img.show()
    return chars


def cut_word_with_size(img: Image.Image, font: str) -> list:
    num_row, num_column = config.DOC_DIM
    w, h = img.size
    w_cut = w // num_column
    h_cut = h // num_row
    img_char_list = []
    for i_y in range(num_row):
        for i_x in range(num_column):
            x1 = i_x * w_cut
            y1 = i_y * h_cut
            x2 = x1 + w_cut
            y2 = y1 + h_cut
            image = img.crop((x1, y1, x2, y2))
            value = pytesseract.image_to_string(image, lang="chi_sim", config="-psm 10")
            char = ChineseCharacter(config.FONT_NAME_Sun, value, image)
            char.save()
            img_char_list.append(char)
        
    return img_char_list


def cut_word_with_size_and_border(output_dir:str, image: Image.Image, font: str, count: int) -> list:
    ret = []
    for row in range(1, config.DOC_DIM[1]-1):
        for col in range(1, config.DOC_DIM[0]-1):
            left = config.BOARDER_SIZE + col*(config.FONT_SIZE+2*config.MARGIN_SIZE)
            upper = config.BOARDER_SIZE + row*(config.FONT_SIZE+2*config.MARGIN_SIZE)
            tmp = image.crop((left,
                              upper,
                              left + config.FONT_SIZE + 2*config.MARGIN_SIZE,
                              upper + config.FONT_SIZE + 2*config.MARGIN_SIZE))
            '''
            tmp = ChineseCharacter(font=font,
                                   value=pytesseract.image_to_string(tmp, lang="chi_sim", config="-psm 10"),
                                   image=tmp)
            '''
            # tmp.show()
            # tmp.save()
            count += 1
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
        img_list = cut_word_with_size_and_border(img, output_folder, 0)
        input_cnt += 1
    print(input_images)
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
    # img.show()
    # img = pretreat.crop_image(img)
    img_list = cut_word_with_size_and_border(img, config.FONT_NAME_HuaWen, 1)
    # print(pytesseract.image_to_string(img_list[0], lang="chi_sim", config="-psm 10"))
    # print(pytesseract.image_to_string(img, lang="chi_sim"))
