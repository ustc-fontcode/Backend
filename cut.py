import sys
sys.path.append("..")
from cutword import cut_word_with_size_and_border
from config import IMAGE_SIZE
from cutword import binaryzation
from PIL import Image 
import os

#   @paramas: 
#   1 -> store input image in input folder
#   2 -> results are stored in output folder

def cutInputImage(input_img_path, output_path, username):
    
    # clean output dir when start cutting
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.system('rm -rf ' + output_path + '/*')
    img = Image.open(input_img_path)
    img = img.resize(IMAGE_SIZE)
    img_list = cut_word_with_size_and_border(output_path, img, username, 0)
    return len(img_list)



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
    input_path = "../output/1.jpg"
    output_path = "../cut"
    # if not os.path.exists(output_path):
    #         os.mkdir(output_path)
    # # os.removedirs("../result")
    # cutTrainImages(input_path, output_path, 2, ["huawen", "micro"])
    cutInputImage(input_path, output_path, '0')
