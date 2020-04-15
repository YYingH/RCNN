import cv2
import os
import random
import shutil
from datasets import Data
from selective_search import selective_search
from utils import read_from_file, convert_to_xywh, crop_image, IOU_calculator, ellipse_to_rectangle

COUNT_FACE, COUNT_BACKGROUND = 0, 0

def generate_selective_search(image_dir):
    img = cv2.imread(image_dir)
    img_lbl, regions = selective_search(
        img, scale=500, sigma=0.8, min_size=10)
    candidates = set()
    for r in regions:
    # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 220 pixels
        if r['size'] < 220:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        if w / h > 2.5 or h / w > 2.5:
            continue
        candidates.add(r['rect'])
    return candidates

def prepare_data(datasets, annotations, threthoud_pos, threthoud_neg, save_path):
    global COUNT_FACE, COUNT_BACKGROUND
    for i in range(len(datasets)):
        image_dir, num_of_faces, gts = datasets[i]
        gts = convert_to_xywh(ellipse_to_rectangle(num_of_faces, gts))

        for gt in gts:
            img = crop_image(image_dir, gt)
            if len(img) == 0:
                continue
            a, b, c = img.shape
            if a == 0 or b == 0 or c == 0:
                continue
            COUNT_FACE += 1
            path = ''.join([save_path, '1/', str(i), '_', str(COUNT_FACE), '.jpg'])
            cv2.imwrite(path, img)

        for candidate in generate_selective_search(image_dir):
            x, y, w, h = candidate
            ious = []
            img = crop_image(image_dir, candidate)
            if len(img) == 0:
                continue
            for gt in gts:
                ious.append(IOU_calculator(x+w/2, y+h/2, w, h,
                    gt[0]+gt[2]/2, gt[1]+gt[3]/2, gt[2], gt[3]))
            if max(ious) > threthoud_pos:
                COUNT_FACE += 1
                path = ''.join([save_path, '1/', str(i), '_', str(COUNT_FACE), '.jpg'])
                cv2.imwrite(path, img)
            elif max(ious) < threthoud_neg:
                COUNT_BACKGROUND += 1
                path = ''.join([save_path, '0/', str(i), '_', str(COUNT_BACKGROUND), '.jpg'])
                cv2.imwrite(path, img)
        print(f"====>>> {i}/{len(datasets)}: Face: {COUNT_FACE}, Background: {COUNT_BACKGROUND}")
    
def dataset_split(path_train_face, path_train_background, path_val):
    face_val, background_val = 0, 0
    for path, dir_list, file_list in os.walk(path_train_face):
        for file_name in file_list:
            if random.randint(0, 10) == 0:
                face_val += 1
                path_ori = os.path.join(path, file_name)
                path_target = path_ori.replace('train/','val/')
                shutil.move(path_ori, path_target)
    for path, dir_list, file_list in os.walk(path_train_background):
        for file_name in file_list:
            if random.randint(0, 200) == 0:
                background_val += 1
                path_ori = os.path.join(path, file_name)
                path_target = path_ori.replace('train/','val/')
                shutil.move(path_ori, path_target)
    print(f'Validation dataset build finished! face: {face_val}, background: {background_val}')



if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    IOU_pos, IOU_neg = 0.7, 0.3

    path_train = ''.join([PROJECT_ROOT, '/data/FDDB_crop/iou_',str(IOU_pos),'/train/'])
    path_val = ''.join([PROJECT_ROOT, '/data/FDDB_crop/iou_',str(IOU_pos),'/val/'])

    for path in [path_train, path_val]:
        for label in ['0/', '1/']:
            if not os.path.exists(path + label):
                os.makedirs(path)

    print("Start to prepare dataset")
    annotations = read_from_file(PROJECT_ROOT + "/data/FDDB/FDDB-folds/")
    datasets = Data(annotations)
    prepare_data(datasets, annotations, threthoud_pos = IOU_pos, threthoud_neg =  IOU_neg, save_path = path_train)
    dataset_split(path_train + '1/', path_train + '0/', path_val)