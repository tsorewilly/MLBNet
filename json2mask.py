#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:51:24 2025

@author: apple
"""

import os, base64, json
from labelme import utils
import numpy as np
from PIL import Image
import imgviz
import PIL.Image

def json_to_mask(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    mask_array = np.array(data['mask'])
    print(mask_array)
    mask_image = Image.fromarray((mask_array*255).astype(np.uint8))
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    
    output_path = os.path.join(output_dir, f"{base_name}.jpg")
    mask_image.save(output_path)
        
    
def convert_json_to_jpg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in os.listdir(input_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(input_dir, json_file) 
            json_to_mask(json_path, output_dir)


def convert_json_to_png(input_dir, output_dir):    
    for json_file in os.listdir(input_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(input_dir, json_file) 
            #print(json_path)
            data = json.load(open(json_path))
            imageData = data.get("imageData")
        
            if not imageData:
                imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
                #print(imagePath)
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {"_background_": 0}
            
            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            lbl, _ = utils.labelme_shapes_to_label(img.shape, data["shapes"])
        
            label_names = [None] * (max(label_name_to_value.values()) + 1)
            
            for name, value in label_name_to_value.items():
                label_names[value] = name
        
            lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(img), label_names=label_names, loc="rb")
            
            filename = str(json_file).split('.')[0]
            filename = output_dir+filename+'.png'
            utils.lblsave(os.path.join(filename), lbl)
            #PIL.Image.fromarray(lbl_viz).save(filename+'_viz.png')
        
    
if __name__ == "__main__":
    src_files = "/Users/apple/Desktop/Projects/ICRA 2025/wSL/Data/Catheter/json/"
    dest = "/Users/apple/Desktop/Projects/ICRA 2025/wSL/Data/Catheter/labels/"
    
    convert_json_to_png(src_files, dest)