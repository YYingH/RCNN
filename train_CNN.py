import os
import torch
import torch.nn as nn
import logging
from utils import load_classify_data
from GoogLeNet import GoogLeNet

Max_acc = 0

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
            if torch.cuda.is_available():
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
        if torch.cuda.is_available():
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
            lr = learning_rate * (0.5 ** (epoch // 20))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    path_train = PROJECT_ROOT + '/data/FDDB_crop/iou_0.5/train/'
    path_val = PROJECT_ROOT + '/data/FDDB_crop/iou_0.5/val/'
    model_path = [PROJECT_ROOT, "/model/","","_GoogLeNet.pt"]

    learning_rate = 1e-4
    logging.basicConfig(level=logging.INFO,filename=PROJECT_ROOT + '/log/GoogLeNet.log',format="%(message)s")
    train_iterator, val_iterator = load_classify_data(path_train, path_val, batch_size = 16, input_size=227)
    model = GoogLeNet(num_classes = 2)
    weights = torch.FloatTensor([1.0,33.5])
    if torch.cuda.is_available():
        model = model.cuda()
        weights = weights.cuda()
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(100):
        train_loss = train(epoch, model, train_iterator,val_iterator, optimizer, criterion, model_path)