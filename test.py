from wordcut.wordcut import cut_word_with_size_and_border
from wordcut.config import *
from PIL import Image 
import os

# @paramas: 
#   1 -> store input image in input folder
#   2 -> cut result are stored in output folder

def cutInputImages(input_path, output_path):
    
    # clean output dir when start cutting
    os.system('rm -rf ./output/*')
    input_images = os.listdir(input_path)
    input_cnt = 0

    for image_name in input_images:
        img = Image.open('./input/' + image_name).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        output_folder = str(input_cnt)
        os.makedirs('./output/' + output_folder)
        img_list = cut_word_with_size_and_border(img, output_folder, 0)
        input_cnt += 1


if __name__ == "__main__":
    input_path = "./input"
    output_path = "./output"
    cutInputImages(input_path, output_path)