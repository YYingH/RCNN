import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn.svm import LinearSVC
from torchvision import transforms
from GoogLeNet import GoogLeNet, Inception, GlobalAvgPool2d, FlattenLayer
from utils import read_from_file
from torchvision import datasets 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

def train_svm(path_pos, path_neg, cnn_model, svm_model):

    model = GoogLeNet(num_classes = 2, is_train = False)
    input_size = 227

    pred_transforms = transforms.Compose([
        transforms.Pad(padding = 16),
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    model.load_state_dict(torch.load(cnn_model, map_location=torch.device('cpu')))
    model.eval()
    
    features = np.array([])
    labels = []
    COUNT_POS = 0
    COUNT_NEG = 0
    # Face
    with torch.no_grad():
        for path,dir_list,file_list in os.walk(path_pos):
            for file_name in file_list:
                if not file_name.endswith('.jpg'):
                    continue
                path_img = os.path.join(path, file_name)
                im = pred_transforms(Image.open(path_img)).unsqueeze_(0)
                featuremap = np.reshape(model(im).numpy(), (1, 1024))
                features = featuremap if features.size == 0 else np.concatenate((features, featuremap))
                labels.append(1)
                COUNT_POS += 1
        

        for path,dir_list,file_list in os.walk(path_neg):
            for file_name in file_list:
                if not file_name.endswith('.jpg'):
                    continue
                if COUNT_NEG == COUNT_POS:
                    break
                path_img = os.path.join(path, file_name)
                im = pred_transforms(Image.open(path_img)).unsqueeze_(0)
                featuremap = np.reshape(model(im).numpy(), (1, 1024))
                features = featuremap if features.size == 0 else np.concatenate((features, featuremap))
                labels.append(0)
                COUNT_NEG += 1

    X_train, X_test, y_train,  y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42)
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'accuracy is: {accuracy_score(y_test, y_pred)*100}')
    joblib.dump(clf, svm_model)




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    path_pos = PROJECT_ROOT + '/data/FDDB_crop/iou_0.7/train/1/'
    path_neg = PROJECT_ROOT + '/data/FDDB_crop/iou_0.7/train/0/'
    train_svm(path_pos, path_neg, cnn_model = 'model/0.9267_GoogLeNet.pt', svm_model = 'model/svm.pkl')