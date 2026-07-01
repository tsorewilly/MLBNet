# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:50:15 2020

@author: Omisore
"""

import labelme
import os, sys

path = os.getcwd()+"/real-time_data/images/"
dest = os.getcwd()+"/real-time_data/labels/"

dirs = os.listdir(path)

i=0

for item in dirs:
    if item.endswith('.json'):        
        if os.path.isfile(path+item):
            my_dest =dest+item[0:-5]#"fin" + str(i)
            #os.system("mkdir "+my_dest)
            if os.system("labelme_json_to_dataset "+dest+item+" -o "+my_dest):
                i=i+1
                print (item + ' was successfully saved to '+my_dest)
            
            
# In[]
import labelme
import os, sys

path='G:/Documents/Asst Prof/Publications/TMI/gw_1841/json/'

dirs = os.listdir(path)

i=0

for item in dirs:
   if item.endswith(".json"):
      if os.path.isfile(path+'/'+item):
         my_dest = path+item[0:-5]
         os.system("mkdir "+my_dest)
         os.system("labelme_json_to_dataset "+item+" -o "+my_dest)
         i=i+1
         
         
         
 