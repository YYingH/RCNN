import os
import cv2
from PIL import Image
from GoogLeNet import GoogLeNet
from sklearn.externals import joblib
from generate_dataset import generate_selective_search, crop_image
from torchvision import transforms
from utils import plt_plot_rectangle
import torch
import numpy as np

def predict(image_dir):
    cnn_model = 'model/0.9267_GoogLeNet.pt'
    model = GoogLeNet(num_classes = 2, is_train = False)
    model.load_state_dict(torch.load(cnn_model, map_location=torch.device('cpu')))
    model.eval()

    clf = joblib.load('model/svm.pkl')

    input_size = 227

    pred_transforms = transforms.Compose([
        transforms.Pad(padding = 16),
        transforms.Resize(size = (input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    for candidate in generate_selective_search(image_dir):
        img = crop_image(image_dir, candidate)
        # convert to PIL format
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im = pred_transforms(img).unsqueeze_(0)
        featuremap = np.reshape(model(im).numpy(), (1, 1024))
        if clf.predict(featuremap) == 1:
            results.append(candidate)
    plt_plot_rectangle(image_dir, results)



if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    predict('img_117.jpg')
                    
    