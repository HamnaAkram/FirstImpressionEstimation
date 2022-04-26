import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from dataset_reader import feat_dataset
from retrival_models import myModel
from torch.optim import SGD, Adam
from torch.nn import L1Loss
import numpy as np
import logging
import matplotlib.pyplot as plt
def train():
    batchsize=128
    train_dataset= feat_dataset(partition="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    validate_dataset = feat_dataset(partition="validate")
    validate_dataloader = DataLoader(validate_dataset, batch_size=batchsize, shuffle=True)

    model=myModel(fusion='linear',dim_embed=256,txt_feat_dim=1024,image_feat_dim=1024,audio_feat_dim=512).to(device=0)

    model.train()

    optimizer=Adam(model.parameters(),lr=0.0001)
    loss_func = L1Loss()
    epochs = 300
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    # correct = total = 0
    # batch_val_loss = 0
    best_acc = 0
    # device =0
    lr = 0.0001
    for epoch in range(epochs):
        logging.info("Epoch : ".format(epoch))
        model.train()
        running_loss = 0.0
        batch_loss = batch_val_loss = batch_val_acc =  batch_acc = 0.0


        for i, data in enumerate(train_dataloader, 0):
            logging.info("Starting Training for Batch")
            v_name, text_feat, audio_feat, img_feat, labels = data

            text_feat = text_feat.float().to(device=0)
            audio_feat = audio_feat.float().to(device=0)
            img_feat = img_feat.float().to(device=0)
            labels = labels.float().to(device=0)

            pred_feats, pred_outputs = model(text_feat, audio_feat, img_feat, labels)
            optimizer = Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()

            ##loss calculation
            loss = loss_func(pred_outputs, labels)
            acc = 1 - loss.item()
            loss.backward()

            # logging.info("Training Loss={}".format(loss.item()))
            # logging.info("Training Accuracy={}".format(acc))
            optimizer.step()
            batch_loss += loss.item()
            batch_acc += 1-loss.item()
            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                      (epoch + 1, i + 1, len(train_dataloader), running_loss / 10))
                running_loss = 0.0

        train_loss.append(batch_loss / len(train_dataloader))
        train_acc.append(batch_acc / len(train_dataloader))
        model.eval()

        # validation
        if len(validate_dataloader)>0:
            with torch.no_grad():
                batch_val_loss = 0.0
                for data in validate_dataloader:
                    v_name, text_feat, audio_feat, img_feat, labels = data
                    optimizer.zero_grad()

                    text_feat = text_feat.float().to(device=0)
                    audio_feat = audio_feat.float().to(device=0)
                    img_feat = img_feat.float().to(device=0)
                    val_labels = labels.float().to(device=0)

                    pred_val_feat, pred_val_outputs = model(text_feat, audio_feat, img_feat, labels)


                    ##loss calculation
                    loss = loss_func(pred_val_outputs, val_labels)
                    acc = 1 - loss.item()
                    # logging.info("Validation Loss={}".format(loss.item()))
                    # logging.info("Validation Accuracy={}".format(acc))
                    batch_val_loss += loss.item()
                    batch_val_acc += 1-loss.item()

                    if acc>best_acc:
                        print("Best validation acc: ", acc)
                        best_acc = acc
                        torch.save(model.state_dict(), "best_model.pth")

                val_loss.append(batch_val_loss / len(validate_dataloader))
                val_acc.append(batch_val_acc/len(validate_dataloader))


            print('Valid accuracy at epoch %d is : %d' % (epoch, (batch_val_acc/len(validate_dataloader))))

        if epoch/5 == 0:
            lr = lr/10

        # save the model
        logging.info("Saving Loss and Accuracy")
        torch.save(model.state_dict(), "/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/model/save_" + str(epoch) + ".pth")
        np.savetxt("/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/train_loss_300_64.csv", train_loss, delimiter=",")
        np.savetxt("/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/train_acc_300_64.csv",train_acc, delimiter=",")
        np.savetxt("/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/val_loss_300_64.csv", val_loss, delimiter=",")
        np.savetxt("/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/val_acc_300_64.csv", val_acc, delimiter=",")



def plot(epoch_loss, train_acc, val_loss, val_acc):


    plt.figure(1)
    #Plotting Values
    plt.plot(epoch_loss, label = "Training Loss")
    plt.plot(val_loss,label = "Validation Loss")
    plt.legend()
    plt.title("Loss Function (MAE)")
    plt.savefig("Loss Function_40_1.png")
    #plt.show()
    plt.figure(2)
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot(train_acc, label="Training Accuracy")
    plt.legend()
    plt.title("Accuracy (1-MAE)")
    plt.savefig("Validation Accuracy_40_1.png")



if __name__=="__main__":

    logging.getLogger().setLevel(level=logging.INFO)
    # train()
    plot(pd.read_csv('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/train_loss_300_64.csv'),
         pd.read_csv('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/train_acc_300_64.csv'),
         pd.read_csv('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/val_loss_300_64.csv'),
         pd.read_csv('/media/hamna/1245D5170555326F/Project: First Impressions/First Impression/FirstImpressionv4/data/result/val_acc_300_64.csv'))

