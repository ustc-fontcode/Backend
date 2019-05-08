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
        os.makedirs(output_path + output_folder)
        img_list = cut_word_with_size_and_border(img, output_folder, 0)
        input_cnt += 1

if __name__ == "__main__":
	hua_path = './HuaWenSun'
	micro_path = './MicroSun'
	huawen = os.listdir(hua_path)
	micro = os.listdir(micro_path)

	os.system('rm -rf '+ hua_path + '_bin'+'/*')
	os.system('rm -rf '+ micro_path + '_bin'+'/*')

	for hua in huawen:
		img = Image.open(hua_path + '/'+hua)
		img = binaryzation(img, 145)
		img.save(hua_path + '_bin' +'/'+ hua)
	
	for mic in micro:
		img = Image.open(micro_path + '/'+mic)
		img = binaryzation(img, 145)
		img.save(micro_path + '_bin' +'/'+ mic)


#     input_path = "./input"
#     output_path = "./output"
#     cutInputImages(input_path, output_path)
 