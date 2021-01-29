import shutil

import tensorflow as tf

import os
from os import listdir
from os.path import isfile, join
from os import walk
from shutil import copyfile
import random
import numpy as np
import io
from object_detection.utils import dataset_util
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import uuid
import glob
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import pandas as pd


class DatasetConverter:
    category_index={}
    categories = []
    image_index =[]

    def __init__(self, categories):        
        self.init_category_index(categories)


    def init_category_index(self, categories):
        self.categories=categories
        self.category_index={}
        num_categories = 0
        #se crea un indice de categorias con id y nombre
        for category in categories:
            if category['enabled']:
                num_categories = num_categories + 1
                category['id'] = num_categories

                self.category_index[category['name']] = category
                for other_name in category['other_names']:
                    self.category_index[other_name] = category


    def label_to_category(self, row_label):
        try:
            return self.category_index[row_label]
        except:
            return None     
        
    


    def get_image(
            self,
            image_path,
            xmins = [],
            xmaxs = [],
            ymins = [],
            ymaxs = [],
            cats = [],
            dataset = None
    ):    
        return {
            'uuid': str(uuid.uuid4()),
            'image_path': image_path, 
            'xmins': xmins, 
            'xmaxs' : xmaxs, 
            'ymins' : ymins, 
            'ymaxs': ymaxs,
            'classes_text':  [category['name'].encode('utf8') for category in cats],
            'classes':  [category['id'] for category in cats],
            'dataset': dataset
        }
    
    

    
    def get_dataset_stadistics(self, images, datasettype):
        df = pd.DataFrame(data = {'image': [image['uuid'] for image in images]})
        for categoy in self.categories:
            df[categoy['name']] = [0 for x in images]
           
        df = df.set_index('image')
        
        for  image in images:
            for col in df.columns: 
                df.at[image['uuid'],col] = df.at[image['uuid'],col] + image['classes_text'].count(col.encode('utf8'))

        df['datasetorigin'] = [ image['dataset'] for image in images]
        df['datasettype'] = [ datasettype for image in images]
        return df
        
    

    def create_labelmap_pbtxt(self, path):
        msg = StringIntLabelMap()
        for category in self.categories:
            if category['enabled']:
                msg.item.append(StringIntLabelMapItem(id=category['id'], name=category['name']))

        txt = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
        with open(path, 'w') as f:
                f.write(txt)
                


    def create_labelmap_txt(self, path):
        f=open(path,'w')
        for category in self.categories:
            if category['enabled']:
                f.write(category['name'] +'\n')
        f.close()




    def load_openImages(self, path, dataset_name = None):
        
        categories_folders = os.listdir(path)
        for category_folder in categories_folders:
            labels =[path + category_folder + "/Label/" + image for  image in os.listdir(path+category_folder+ "/Label")]
            
            for label_path in labels:          
                image_id = label_path.split("/")[-1].split(".txt")[0]  
                image_path= os.path.join(path, category_folder,  image_id + '.jpg')

                annotations_txt = open(label_path, "r").readlines()
                anot_list = [(' '.join(a.split(" ")[0:-4]), list(map(float, a.replace("\n", "").split(" ")[-4:]))) for a in annotations_txt]
        
                xmins = []
                xmaxs = []
                ymins = []
                ymaxs = []
                cats = []
                for _tuple in anot_list:
                    category = self.label_to_category(self.category_index, _tuple[0])                
                    
                    if (category != None):                
                        xmins.append(_tuple[1][0] )
                        ymins.append(_tuple[1][1] )
                        xmaxs.append(_tuple[1][2] )
                        ymaxs.append(_tuple[1][3] )
                        cats.append(category)

                        
                if (len(cats)>0):
                    element  = self.get_image( image_path, xmins, xmaxs, ymins, ymaxs, cats, dataset_name)
                    self.image_index.append(element)

       
    
    
def load_pascalVOC(self, labels_path, images_path, dataset_name = None):
    for xml_file in glob.glob(labels_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        
        image_path =  os.path.join(images_path, filename)
        
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        cats = []
        
        for member in root.findall('object'):
            label = member[0].text
            category = self.label_to_category(self.category_index, label)
            if (category != None): 
                bndbox = member.findall("bndbox")[0]
                xmins.append(int(bndbox.findall('xmin')[0].text) )
                ymins.append(int(bndbox.findall('ymin')[0].text) )
                xmaxs.append(int(bndbox.findall('xmax')[0].text) )
                ymaxs.append(int(bndbox.findall('ymax')[0].text ) )
                cats.append(category)
                
                
        if (len(cats) > 0):
            element=self.get_image( image_path, xmins, xmaxs, ymins, ymaxs, cats, dataset_name)
            self.image_index.append(element)               





    def create_group(self, path, name, images):
        directory_path = os.path.join(path, name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            
            
        writer = tf.io.TFRecordWriter(os.path.join(directory_path,  name + '.record'))    
        
        for info_image in images:
            image_id = info_image['uuid']
            new_image_path= os.path.join(directory_path,  str(image_id) + '.jpg')
            
            copyfile(info_image['image_path'], new_image_path)
            
            with tf.io.gfile.GFile(new_image_path, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            width, height = image.size
            
            xmins = [number / width for number in info_image['xmins']]
            xmaxs = [number / width for number in info_image['xmaxs']]
            ymins = [number / height for number in info_image['ymins']]
            ymaxs = [number / height for number in info_image['ymaxs']]
            classes_text = info_image['classes_text']
            classes = info_image['classes']
            
            filename = new_image_path.encode('utf8')
            image_format = b'jpg'
            tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(filename),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(image_format),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            writer.write(tf_example.SerializeToString())
        
        writer.close()
            

        

    def store_dataset(self, path_to_store, train_ratio =.8, validation_ratio=.10, test_ratio=.10):
        random.shuffle(image_index)

        train_size = int(train_ratio * len(image_index))
        val_size = int(validation_ratio * len(image_index))
        test_size = int(test_ratio * len(image_index))

        x_train = image_index[:train_size]
        x_val = image_index[len(x_train):len(x_train)+val_size]
        x_test = image_index[len(x_train) + len(x_val):]

        
        create_group(path_to_store, 'train', x_train)
        create_group(path_to_store, 'validation', x_val)
        create_group(path_to_store, 'test', x_test)

        df1 = self.get_dataset_stadistics(x_train, 'train')
        df2 = self.get_dataset_stadistics(x_val, 'validation')
        df3 = self.get_dataset_stadistics(x_test, 'test')

        return pd.concat([df1, df2, df3])
    