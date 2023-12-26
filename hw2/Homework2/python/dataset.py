import os
import cv2
import glob

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def load_data(): 
    data = {'color': {}, 'gray': {}}
    
    for i in range(1, 5):
        # glob to find all files in a directory
        path = os.path.join(CURRENT_DIR, '..', f'data{i}', '*')
        images = glob.glob(path)
        images_name = [os.path.basename(img) for img in images]
        data['color'][f'data{i}'] = dict(zip(images_name, [cv2.imread(img) for img in images]))
        data['gray'][f'data{i}'] = dict([(name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for name, img in data['color'][f'data{i}'].items()])
        
    return data

def makedirs(_path):
    path = os.path.join(CURRENT_DIR, _path)
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == "__main__":
    load_data()