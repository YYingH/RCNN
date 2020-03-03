import cv2
import logging
import torch
import os
import torch.nn as nn
import pickle
from datasets import Data
from utils import convert_to_xywh, ellipse_to_rectangle, IOU_calculator, crop_image
from utils import read_from_file, load_classify_data
from selective_search import selective_search
from GoogLeNet import GoogLeNet

Max_acc = 0
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


def prepare_data(datasets, annotations, threthoud = 0.5, save_path = 'dataset.pkl'):
    global COUNT_FACE, COUNT_BACKGROUND
    images, labels = [], []
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
            images.append(img)
            labels.append(1)

        for candidate in generate_selective_search(image_dir):
            x, y, w, h = candidate
            ious = []
            img = crop_image(image_dir, candidate)
            if len(img) == 0:
                continue
            for gt in gts:
                ious.append(IOU_calculator(x+2/w, y+h/2, w, h,
                    gt[0]+gt[2]/2, gt[1]+gt[3]/2, gt[2], gt[3]))
            if max(ious) >= threthoud:
                COUNT_FACE += 1
                images.append(img)
                labels.append(1)
            else:
                COUNT_BACKGROUND += 1
                images.append(img)
                labels.append(0)
        print(f"====>>> {i}/{len(datasets)}: Face: {COUNT_FACE}, Background: {COUNT_BACKGROUND}")
    pickle.dump((images, labels), open(save_path, 'wb'))
    
        


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
        
def train(epoch, model, train_iterator, val_iterator, optimizer, criterion, model_path):
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
                model_path[2] = "{:.4f}".format(test_acc)
                Max_acc = test_acc
                torch.save(model.state_dict(), "".join(model_path))
        # VGG
        if epoch>0 and epoch%20 == 0:
            lr = learning_rate * (0.5 ** (epoch // 30))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    data_is_prepared = False
    classify_model = True
    path_train = PROJECT_ROOT + '/data/FDDB_crop/train/'
    path_test = PROJECT_ROOT + '/data/FDDB_crop/test/'

    if data_is_prepared == False:
        print("Start to prepare dataset")
        annotations = read_from_file(PROJECT_ROOT + "/data/FDDB/FDDB-folds/")
        datasets = Data(annotations)
        prepare_data(datasets, annotations, threthoud = 0.5, save_path = 'dataset.pkl')
    
    if classify_model == False:
        model_path = [PROJECT_ROOT, "/model/","","_GoogLeNet.pt"]
        learning_rate = 1e-4
        logging.basicConfig(level=logging.INFO,filename=PROJECT_ROOT + '/log/GoogLeNet.log',format="%(message)s")
        train_iterator, val_iterator = load_classify_data(path_train, path_test, batch_size = 32, input_size=227)
        model = GoogLeNet(num_classes = 2)
        is_cuda = torch.cuda.is_available()
        weights = torch.FloatTensor([19.33,1.0])
        if is_cuda:
            model = model.cuda()
            weights = weights.cuda()
        criterion = nn.CrossEntropyLoss(weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(100):
            train_loss = train(epoch, model, train_iterator,val_iterator, optimizer, criterion, model_path)


