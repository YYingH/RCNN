import cv2
import logging
import torch
import torch.nn as nn
import numpy as np
from datasets import Data
from PIL import Image
from utils import convert_to_xywh, plt_plot_rectangle, ellipse_to_rectangle, IOU_calculator, crop_image
from utils import read_from_file, load_classify_data
from selective_search import selective_search
from GoogLeNet import GoogLeNet

Max_acc = 0

def generate_selective_search(img):
    img_lbl, regions = selective_search(
        img, scale=1.0, sigma=0.8, min_size=50)
    candidates = set()
    for r in regions:
    # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 1000 pixels
        if r['size'] < 2500:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 2.5 or h / w > 2.5:
            continue
        candidates.add(r['rect'])
    return candidates


def prepare_data(image, gts, path_train, path_test):
    count_pos, count_neg = 0, 0
    for gt in gts:
        img = crop_image(image, gt)
        cv2.imwrite(path_train + str(count_pos) + '.jpg', img)
        count_pos += 1
    for candidate in generate_selective_search(image):
        x, y, w, h = candidate
        ious = []
        for gt in gts:
            ious.append(IOU_calculator(x+2/w, y+h/2, w, h,
                gt[0]+gt[2]/2, gt[1]+gt[3]/2, gt[2], gt[3]))
        iou = max(ious)
        if iou >= 0.65:
            img = crop_image(image, candidate)
            cv2.imwrite(path_train + str(count_pos) + '.jpg', img)
            count_pos += 1
        elif iou == 0:
            img = crop_image(image, candidate)
            cv2.imwrite(path_test + str(count_neg) + '.jpg', img)
            count_neg += 1


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def evaluate(epoch, model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            if is_cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            fx = model(x)
            loss = criterion(fx, y)
            acc = calculate_accuracy(fx, y)
            epoch_loss += loss.item()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(f"Test: [{epoch}] >>> the accuracy is [{epoch_acc / len(iterator):.4f}]")       
    logging.info(f"Test: [{epoch}] >>> the accuracy is [{epoch_acc / len(iterator):.4f}]")           
    return epoch_acc / len(iterator)
        
def train(epoch, model, train_iterator, val_iterator, optimizer, criterion):
    global Max_acc
    epoch_loss, epoch_acc = 0, 0 
    i = 0
    model.train()
    for(x, y) in train_iterator:
        if is_cuda:
            x,y = x.cuda(), y.cuda()
        i += 1
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        print(f"Train: Epoch: [{epoch}] >>> ({i}/{len(train_iterator)}), and the current loss is [{loss:.4f}]") 
        logging.info(f"Train: Epoch: [{epoch}] >>> ({i}/{len(train_iterator)}), and the current loss is [{loss:.4f}]") 
        
        if i % 40 == 0:
            test_acc = evaluate(epoch, model, val_iterator, optimizer, criterion)
            if test_acc > Max_acc:
                model_path[1] = "{:.4f}".format(test_acc)
                Max_acc = test_acc
                torch.save(model.state_dict(), "".join(model_path))
        # VGG
        if epoch>0 and epoch%20 == 0:
            lr = learning_rate * (0.5 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    



if __name__ == "__main__":
    data_is_prepared = True
    classify_model = False
    annotations = read_from_file("data/FDDB/FDDB-folds/")
    datasets = Data(annotations)
    path_train = 'data/FDDB_crop/train/'
    path_test = 'data/FDDB_crop/test/'
    
    if data_is_prepared == False:
        for i in range(len(datasets)):
            image, image_dir, num_of_faces, gt_box = datasets[i]
            gt_box = convert_to_xywh(image, ellipse_to_rectangle(num_of_faces, gt_box))
            # plt_plot_rectangle(image, gt_box)
            # candidates = generate_selective_search(image, gt_box)
            # plt_plot_rectangle(image, candidates)
            prepare_data(image, gt_box, path_train + str(i) + '_', path_test + str(i) + '_')
    
    if classify_model == False:
        model_path = ["model/","","_GoogLeNet.pt"]
        learning_rate = 1e-4
        logging.basicConfig(level=logging.INFO,filename='log/GoogLeNet.log',format="%(message)s")
        train_iterator, val_iterator = load_classify_data(path_train, path_test, batch_size = 32, input_size=224)
        model = GoogLeNet(num_classes = 2)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            model = model.cuda()
        weights = torch.FloatTensor([19.33,1.0])
        criterion = nn.CrossEntropyLoss(weights)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(100):
            train_loss = train(epoch, model, train_iterator,val_iterator, optimizer, criterion)


