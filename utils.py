""" This file for data loading, containing a variety of useful functions
In this file:

function:


FDDB annotation structure:
    image path
    num of faces
    major_axis_radius     minor_axis_radius       angle           center_x        center_y 1
"""
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,WeightedRandomSampler


def cv2_plot_ellipse(img, n, locations):
    """ Draw elliptical boxes on the image """
    major_axis_radius, minor_axis_radius, angle, center_x, center_y = locations
    for i in range(n):
        cv2.ellipse(img, center = (int(center_x[i]), int(center_y[i])), 
        axes = (int(minor_axis_radius[i]), int(major_axis_radius[i])),
        angle = angle[i], startAngle = 0, endAngle = 360, color = 255)
    cv2.imshow('image', img)
    cv2.waitKey(0)


def cv2_plot_rectangle(img, n, pts):
    """ Draw rectangular boxes on the image """
    for pt in pts:
        cv2.rectangle(img, pt1=pt[0], pt2=pt[1], color=(255, 0, 255))
    cv2.imshow('image', img)
    cv2.waitKey(0)


def plt_plot_rectangle(img, locations):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for x, y, w, h in locations:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def ellipse_to_rectangle(n, locations):
    """ Change elliptical boxes information into rectanglar boxes """
    major_axis_radius, minor_axis_radius, angle, center_x, center_y = locations
    pt = []
    for i in range(n):
        pt1 = (int(center_x[i]) - int(minor_axis_radius[i]), int(center_y[i]) - int(major_axis_radius[i]))
        pt2 = (int(center_x[i]) + int(minor_axis_radius[i]), int(center_y[i]) + int(major_axis_radius[i]))
        pt.append([pt1, pt2])
    return pt


def convert_to_xywh(image, pts):
    """ convert coordinate to xywh format """
    boxes = []
    for pt1, pt2 in pts:
        x = pt1[0]
        y = pt1[1]
        w = pt2[0] - pt1[0]
        h = pt2[1] - pt1[1]
        boxes.append([x, y, w, h])
    return boxes


def read_from_file(path = "data/FDDB/FDDB-folds/"):
    files = [path+file for file in os.listdir(path) if file.endswith("-ellipseList.txt")]
    annotations = []
    for file in files:
        img_annot = []
        i = 0
        for line in open(file).readlines():
            line = line.replace("\n","")
            i += 1 
            if "img" in line and i>1:
                annotations.append(img_annot)
                img_annot = []
            img_annot.append(line)
        annotations.append(img_annot)
    return annotations


def IOU_calculator(x1, y1, w1, h1, x2, y2, w2, h2):
    """ 
    Computing IOU
    param: (x1, y1) and (x2, y2) are the center point of the two rectangular boxes
    """
    iou = 0
    if((abs(x1 - x2) < ((w1 + w2)/ 2.0)) and (abs(y1-y2) < ((h1 + h2)/2.0))):
        left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
        upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))
      
        right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
        bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))
        
        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        union_square = (w1 * h1)+(w2 * h2)-inter_square
        
        iou = inter_square/union_square * 1.0
    return iou

def crop_image(img, gt):
    x, y, w, h = gt
    return img[y:y+h, x:x+w]



def load_classify_data(train_dir, val_dir, batch_size, input_size = 224):
    train_transforms = transforms.Compose([
        transforms.Pad(padding = 16),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Pad(padding = 16),
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    

    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size = batch_size, shuffle=True, num_workers=2)

    val_datasets = datasets.ImageFolder(val_dir, val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size = batch_size, shuffle = True, num_workers = 2)

    return train_dataloader, val_dataloader

