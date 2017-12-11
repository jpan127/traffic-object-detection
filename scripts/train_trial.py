import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = [     
    'DontCare'             ,
    'Car'                  ,
    'SUV'                  ,
    'SmallTruck'           ,
    'MediumTruck'          ,
    'LargeTruck'           ,
    'Pedestrian'           ,
    'Bus'                  ,
    'Van'                  ,
    'GroupOfPeople'        ,
    'Bicycle'              ,
    'Motorcycle'           ,
    'TrafficSignal-Green'  ,
    'TrafficSignal-Yellow' ,
    'TrafficSignal-Red'    ,
]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_annotation(tmp,filename):
    in_file = open(filename)
    new_path = filename.replace('data_training','files_darknet_trial_final')
    new_path = new_path.replace('xml','txt')
    print new_path
    #out_file = open('/home/dnn_benchmark/dataset/files_darknet_trial/' +  tmp +'.txt', 'w+')
    directory = os.path.dirname(new_path)
    if not os.path.exists(directory):
    	os.makedirs(directory)    
    out_file = open(new_path, 'w');
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
#    if os.stat('/home/dnn_benchmark/dataset/data_training/files_darknet/'+ tmp + '.txt').st_size == 0: 
#        os.remove('/home/dnn_benchmark/dataset/data_training/files_darknet/'+ tmp + '.txt')
#         print tmp

root_dir = r'/home/dnn_benchmark/dataset/data_training'
global train_file
train_file = open("train.txt",'wb')
for root,dirs,files in os.walk(root_dir):
       for filename in files:
           if filename.endswith(".xml"):
                 tmp = filename
                 tmp = tmp.split('.')[0]
                 convert_annotation(tmp,root+'/'+filename)
                 train_file.write(root + '/' + tmp + '.jpg' + '\n')
train_file.close()
#                 break
#                 print  filename
