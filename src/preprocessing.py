from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from PIL import Image
import pandas as pd
import glob
import os

# Config for input folders
class CONFIG:
    ROOT_DIR   = './' #path to th dataset root
    DATA_DIR   = os.path.join(ROOT_DIR, 'face-mask-detection')
    IMAGE_DIR  = os.path.join(DATA_DIR, 'images')
    LABEL_DIR  = os.path.join(DATA_DIR, 'annotations')
    
    OUTPUT_DIR = '../datasets'
    
    
    LABEL_MAP = {'without_mask'         : 0,
                 'with_mask'            : 1,
                 'mask_weared_incorrect': 2}

class Preprocess:
    def __init__(self):
        # Create folders as per yolov8 format for saving training data
        os.mkdir('../datasets')
        os.mkdir('../datasets/face-mask-detection')
        os.mkdir('../datasets/face-mask-detection/images')
        os.mkdir('../datasets/face-mask-detection/labels')

        os.mkdir('../datasets/face-mask-detection/images/train')
        os.mkdir('../datasets/face-mask-detection/labels/train')

        os.mkdir('../datasets/face-mask-detection/images/val')
        os.mkdir('../datasets/face-mask-detection/labels/val')

        os.mkdir('../datasets/face-mask-detection/images/test')
        os.mkdir('../datasets/face-mask-detection/labels/test')
        
    def copy_image_file(self, image_items, folder_name):
        for image in image_items:
            img = Image.open(f'{CONFIG.IMAGE_DIR}/{image}')
            img1 = img.resize((640, 480))
            _ = img1.save(f'{CONFIG.OUTPUT_DIR}/face-mask-detection/images/{folder_name}/{image}')
    
    def copy_label(self, df, img_file_path, folder_name):
        file_name = [x.split('.')[0] for x in img_file_path]
        for name in file_name:
            data = df[df.name == name]

            box_list = []
            for idx in range(len(data)):
                row = data.iloc[idx]
                box_list.append(row['class']+" "+row['x_center']+" "+row['y_center']+" "+ row['box_width']+" "+row['box_height'])

            text = "\n".join(box_list)
            with open(f'{CONFIG.OUTPUT_DIR}/face-mask-detection/labels/{folder_name}/{name}.txt', 'w') as file:
                file.write(text)
    
    def create_yaml(self):
        # Configure .yaml file for yolov8
        yaml_file = """path: ./face-mask-detection
        train: images/train
        val: images/val
        test: images/test
                        
        nc: 3
        names: 
        0: without_mask
        1: with_mask
        2: mask_weared_incorrect"""

        with open('../datasets/face-mask-detection/data.yaml', 'w') as f:
            f.write(yaml_file)
    
    def extract_annotation(self):
        # Extract annotation from xml file
        df = {
            'name': [],
            'label': [],
            'width': [],
            'height': [],
            'xmin': [],
            'ymin': [],
            'xmax': [],
            'ymax': []
            }

        for _, anno in enumerate(glob.glob(CONFIG.LABEL_DIR + '/*.xml')):
            trees = ET.parse(anno)
            
            root = trees.getroot()
            width, height = [], []
            for item in root.iter():
                    
                if item.tag == 'size':
                    for attr in list(item):
                        if attr.tag == 'width':
                            width = int(round(float(attr.text)))
                        if attr.tag == 'height':
                            height = int(round(float(attr.text)))
                            
                if item.tag == 'object':
                    for attr in list(item):
                        if 'name' in attr.tag:
                            label = attr.text
                            df['label'] += [label]
                            df['width'] += [width]
                            df['height'] += [height]
                            df['name'] += [anno.split('/')[-1][0:-4]]
                            
                        if 'bndbox' in attr.tag:
                            for dim in attr:
                                if dim.tag == 'xmin':
                                    xmin = int(round(float(dim.text)))
                                    df['xmin'] += [xmin]
                                    
                                if dim.tag == 'ymin':
                                    ymin = int(round(float(dim.text)))
                                    df['ymin'] += [ymin]
                                if dim.tag == 'xmax':
                                    xmax = int(round(float(dim.text)))
                                    df['xmax'] += [xmax]
                                if dim.tag == 'ymax':
                                    ymax = int(round(float(dim.text)))
                                    df['ymax'] += [ymax]
        return pd.DataFrame(df)
    
    def main(self):
        df = self.extract_annotation()
        
        # make img_file_path
        img_file_path = []

        for img in os.listdir(CONFIG.IMAGE_DIR):
            img_file_path.append(f'{img}')

        # split train, test, val data
        train, test = train_test_split(img_file_path, test_size=0.2, random_state=101)
        train, val = train_test_split(train, test_size=0.15, random_state=101)

        # copy the image data in the yolov8 folder 
        self.copy_image_file(train, 'train')
        self.copy_image_file(val, 'val')
        self.copy_image_file(test, 'test')

        df['class'] = df['label'].map(CONFIG.LABEL_MAP)
        df['xmin'] = (640/df['width']) * df['xmin']
        df['ymin'] = (480/df['height']) * df['ymin']
        df['xmax'] = (640/df['width']) * df['xmax']
        df['ymax'] = (480/df['height']) * df['ymax']

        df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].astype('int')

        WIDTH = 640
        HEIGHT = 480

        df['x_center'] = (df['xmin'] + df['xmax']) / (2*WIDTH)
        df['y_center'] = (df['ymin'] + df['ymax']) / (2*HEIGHT)
        df['box_width'] = (df['xmax'] - df['xmin']) / WIDTH
        df['box_height'] = (df['ymax'] - df['ymin']) / HEIGHT

        df = df.astype(str)


        self.copy_label(df, img_file_path, 'train')
        self.copy_label(df, img_file_path, 'val')
        self.copy_label(df, img_file_path, 'test')

        self.create_yaml()
if __name__ == "__main__":
    Preprocess().main()
    