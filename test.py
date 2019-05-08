from wordcut.wordcut import cut_word_with_size_and_border
from wordcut.config import *
from wordcut.wordcut import binaryzation
from PIL import Image 
import os

# @paramas: 
#   1 -> store input image in input folder
#   2 -> cut result are stored in output folder

def cutInputImages(input_path, output_path):
    
    # clean output dir when start cutting
    os.system('rm -rf ' + output_path + '*')
    input_images = os.listdir(input_path)
    input_cnt = 0

    for image_name in input_images:
        img = Image.open(input_path + image_name).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        output_folder = str(input_cnt)
        os.makedirs('./output/' + output_folder)
        img_list = cut_word_with_size_and_border(output_path, img, output_folder, 0)
        input_cnt += 1



# input_path: exp ./photo/normal/ 
# output_path: exp ./result/
# fontnum: exp 2
# font_name:  exp ["HuaWenSun", "MicroSun"]
def cutTrainImages(input_path, output_path, fonts_num, fonts_name):

        # clean output_dir
        os.system("rm -rf {}/*".format(output_path))

        for font_name in fonts_name:

                # font_dir exp photo/normal/HuaWenSun
                font_dir = "{}/{}".format(input_path, font_name)
                # makedir results/normal/HuaWenSun
                os.makedirs("{}/{}".format(output_path, font_name))

                image_cnt = 0
                images = os.listdir(font_dir)
                for image in images:
                        img = Image.open("{}/{}".format(font_dir, image))
                        img = img.resize(IMAGE_SIZE)
                        img_list = cut_word_with_size_and_border(output_path, img, font_name, image_cnt)
                        image_cnt += len(img_list)



if __name__ == "__main__":
    input_path = "../photo/normal"
    output_path = "../results/normal"
    cutTrainImages(input_path, output_path, 2, ["HuaWenSun", "MicroSun"])
