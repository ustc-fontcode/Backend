from wordcut.wordcut import cut_word_with_size_and_border
from wordcut.config import *
from wordcut.wordcut import binaryzation
from PIL import Image 
import os

# @paramas: 
#   1 -> store input image in input folder
#   2 -> cut results are stored in output folder

def cutInputImages(input_path, output_path):
    
    # clean output dir when start cutting
    os.system('rm -rf ' + output_path + '/*')
    input_images = os.listdir(input_path)
    input_cnt = 0

    for image_name in input_images:
        img = Image.open("{}/{}".format(input_path,image_name)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        cut_folder = str(input_cnt)
        os.makedirs("{}/{}".format(output_path, cut_folder))
        img_list = cut_word_with_size_and_border(output_path, img, cut_folder, 0)
        input_cnt += 1



# input_path: exp ./photo/normal/ 
# output_path: exp ./result/
# fontnum: exp 2
# font_name:  exp ["HuaWenSun", "MicroSun"]
# clean output_dir
def cutTrainImages(input_path, output_path, fonts_num, fonts_name, cnt = 0):

        for font_name in fonts_name:

                # font_dir exp photo/normal/HuaWenSun
                font_dir = "{}/{}".format(input_path, font_name)
                # makedir results/normal/HuaWenSun
                if not os.path.exists("{}/{}".format(output_path, font_name)):
                        os.makedirs("{}/{}".format(output_path, font_name))
                image_cnt = cnt
                images = os.listdir(font_dir)
                for image in images:
                        img = Image.open("{}/{}".format(font_dir, image))
                        img = img.resize(IMAGE_SIZE)
                        img_list = cut_word_with_size_and_border(output_path, img, font_name, image_cnt)
                        img.load()
                        image_cnt += len(img_list)        
        return image_cnt



if __name__ == "__main__":
    input_path = ["../thresh"]
    output_path = "../results"
    os.system("rm -rf {}/*".format(output_path))
    cnt = 0
    for i in input_path:
        print(i)
        cnt = cutTrainImages(i, output_path, 2, ["HuaWenSun", "MicroSun"], cnt)
