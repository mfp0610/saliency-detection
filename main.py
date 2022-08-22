import os
import numpy as np
import random
import cv2
from PIL import Image

from utils.datasets import ImageDataset
from demo.itti import ITTISaliencyMap
from demo.mdc import MDCSaliency
 

if __name__ == "__main__":
    root = './dataset/SALICON/'
    dataset = ImageDataset(root)

    # random 10 images
    id_list = [random.randint(0,len(dataset))for i in range(10)]

    for id in id_list :
        img, map = dataset[id]
        
        # using ITTI algorithm
        saliency = ITTISaliencyMap(np.asarray(img))
        img_out = Image.fromarray(np.uint8(255 * saliency.map))

        output_dir = "./output/itti/"
        os.makedirs(output_dir, exist_ok=True)
        img.save(output_dir + "itti" + str(id) + "_img.jpg")
        map.save(output_dir + "itti" + str(id) + "_map.jpg")
        img_out.save(output_dir + "itti" + str(id) + "_res.jpg")

        # using MDC algorithm
        img_ = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) 
        saliency = MDCSaliency(img_)
        img_out = [ Image.fromarray(np.uint8(saliency.original_map)), Image.fromarray(np.uint8(saliency.aug_map)) ]

        output_dir = "./output/mdc/"
        os.makedirs(output_dir, exist_ok=True)
        img.save(output_dir + "mdc" + str(id) + "_img.jpg")
        map.save(output_dir + "mdc" + str(id) + "_map.jpg")
        img_out[0].save(output_dir + "mdc" + str(id) + "_res.jpg")
        img_out[1].save(output_dir + "mdc" + str(id) + "_res_enh.jpg")