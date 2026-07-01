import os
import numpy as np
import pandas as pd
import random 
import shutil


#_dir = "./gw_1841/"
_dir = "G:/Documents/Post-Doc/USA/UPenn/Project/Phuong/data_cardiac/images/"
_dir1 = "G:/Documents/Post-Doc/USA/UPenn/Project/Phuong/data_cardiac/labels/"
_dir2 = "G:/Documents/Post-Doc/USA/UPenn/Project/Phuong/data_cardiac/"

start = 1001
pt=0

files = [f.name for f in os.scandir(_dir) if not f.is_dir()] 
np.random.shuffle(files)

trainlabels = files[:int(len(files)*0.8)]

with open(_dir2+"gluteus_medius.txt","w") as f:
    for file in files:
        nname = str(start)+"_"+str(pt)
        
        #shutil.move(_dir+file, _dir+nname+"_type-gray.nii.gz")
        #shutil.move(_dir1+file.split(".")[0]+"_gt.nii.gz", _dir1+nname+"_type-seg.nii.gz")
        pt =  np.abs(pt-1)
        
        start+=1
        
        if file in trainlabels:
            f.write(nname+"\n")    

# In[]
"""
np.random.shuffle(files)

with open(_dir2+"gluteus_medius.txt","w") as f:
#Generate file of partitioned training and testing set
    for file in files[:int(len(files)*0.4)]:
        f.write(file.split(".")[0]+"\n")    
    

with open(_dir+"test.txt",'w') as f:
    for file in files[int(len(files)*0.9):]:
        f.write(file.split('.')[0]+"\n")    
"""        