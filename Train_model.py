


import torch
import random
import os
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import warnings
from SimilarityModelDeepStride64 import SimilarityNetwork,TripletLoss, SimilarityNetworkDataset
from shutil import copyfile
from datetime import date


warnings.filterwarnings("ignore")



class Config():
    train_batch_size = 16 # V1,V2 = 16
    train_number_epochs = 20 #50
    lrate = 0.00001 #V1-V4:0.00005
    folder_dataset = r'C:\Users\shelly_levi\Documents\similarity\Results\09_09_22'
    print_every = 200 
    
    today = date.today()
    date_str = today.strftime("%m_%d_%y")
    folder_dataset_train = folder_dataset + '\Training'
    folder_dataset_dev = folder_dataset + '\Validation'
    folder_model_main = folder_dataset +  '\\' + date_str 
    folder_model = folder_model_main 
    if not os.path.isdir(folder_model):
        os.mkdir(folder_model)

    
def save_plot_loss(train_loss_history,dev_loss_history,epoch,save_folder,best_loss_epoch):
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve - loss")
    plt.plot(train_loss_history, label="Train loss")
    plt.plot(dev_loss_history, label="Val loss")
    plt.plot(best_loss_epoch, dev_loss_history[best_loss_epoch], marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.ylim(0,6)
    plt.legend();
    plt.grid()
    plt.savefig(save_folder + '\\Learning curve - loss.jpg')
     
def save_plot_acc(train_acc_history,dev_acc_history,epoch,save_folder,best_loss_epoch):
    
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve - acc")
    plt.plot(train_acc_history, label="Train acc")
    plt.plot(dev_acc_history, label="Val acc")
    plt.plot(best_loss_epoch, dev_acc_history[best_loss_epoch], marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy [%]")
    plt.ylim(0,100)
    plt.legend();
    plt.grid()
    plt.savefig(save_folder + '\\Learning curve - accuracy.jpg')    
    
if __name__ == "__main__":  

    Num_gpu = 0
    num_workers = 0
    best_loss_epoch = 1000
    lrate = Config.lrate
    weight_decay = 0.5
    
    net = SimilarityNetwork().cuda()
    criterion = TripletLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(),lr = lrate,weight_decay = weight_decay) #, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    Net_config = {'Learning_rate':lrate,'Batch_size':Config.train_batch_size,'Num_epochs': \
                  Config.train_number_epochs}
    torch.save(Net_config, Config.folder_model + '\\' + 'NetConfig.p')
    
    Model_file = 'SimilarityModelDeepStride.py'
    src = os.getcwd() + '\\' + Model_file
    dst = Config.folder_model + '\\' + Model_file
    copyfile(src, dst)

    Train_file = 'Train_model.py'
    src = os.getcwd() + '\\' + Train_file
    dst = Config.folder_model + '\\' + Train_file    
    copyfile(src, dst)
        
    counter = []
    dis_pos_history = []
    dis_neg_history = []
    train_loss_history = []
    train_acc_history = []
    dev_loss_history = []
    dev_acc_history = []

    iteration_number= 0
    Similarity_dataset_train = SimilarityNetworkDataset(imageFolderDataset=Config.folder_dataset_train)
    Trainloader = DataLoader(Similarity_dataset_train,num_workers=num_workers,batch_size=Config.train_batch_size,shuffle=True)
    
    Similarity_dataset_dev = SimilarityNetworkDataset(imageFolderDataset=Config.folder_dataset_dev)
    DevLoader = DataLoader(Similarity_dataset_dev,num_workers=0,batch_size=1,shuffle=True) #=Num_gpu*4
    for epoch in range(0,Config.train_number_epochs):
        train_epoch_loss = []
        train_epoch_acc = []
        net.train()
        for i, data in enumerate(Trainloader,0):
            img_anchor, img_pos, img_neg = data
            
            img_anchor, img_pos, img_neg = Variable(img_anchor).cuda(), Variable(img_pos).cuda() , Variable(img_neg).cuda()
            output_anchor = net(img_anchor)
            output_pos = net(img_pos)
            output_neg = net(img_neg)

            optimizer.zero_grad()
            loss_contrastive,dis_pos,dis_neg,acc = criterion(output_anchor,output_pos,output_neg)
            loss_contrastive.backward()
            optimizer.step()
            Bscans_num = (i+1)*Config.train_batch_size
            Total_Bscans = len(Trainloader)*Config.train_batch_size
            train_epoch_loss.append(loss_contrastive.item())
            train_epoch_acc.append(acc.item())
            
            if i%Config.print_every==0:
                print("Epoch {} : {}\{}, train loss = {:.2f}, train acc = {:.2f}".format(epoch,Bscans_num,Total_Bscans,train_epoch_loss[-1],train_epoch_acc[-1]))
            

        train_loss = sum(train_epoch_loss)/len(train_epoch_loss)
        train_loss_history.append(train_loss)
        train_acc = sum(train_epoch_acc)/len(train_epoch_acc)*100
        train_acc_history.append(train_acc)
        
        dev_epoch_loss = []
        dev_epoch_acc = []
        net.eval()
        for i, data in enumerate(DevLoader,0):
            img_anchor, img_pos, img_neg = data 
            img_anchor, img_pos, img_neg = Variable(img_anchor).cuda(), Variable(img_pos).cuda() , Variable(img_neg).cuda()
            output_anchor = net(img_anchor)
            output_pos = net(img_pos)
            output_neg = net(img_neg)
            loss_contrastive_dev,dis_pos,dis_neg,acc = criterion(output_anchor,output_pos,output_neg)
            dev_epoch_acc.append(acc.item())
            dev_epoch_loss.append(loss_contrastive_dev.item())

        dev_loss = sum(dev_epoch_loss)/len(dev_epoch_loss)
        scheduler.step(dev_loss)
        dev_loss_history.append(dev_loss)
        dev_acc = sum(dev_epoch_acc)/len(dev_epoch_acc)*100
        dev_acc_history.append(dev_acc)
        
        if dev_loss<best_loss_epoch:
            best_loss_epoch = dev_loss
            best_epoch = epoch
            torch.save(net.state_dict(), Config.folder_model + '\\' + 'BestDev.p')
                       
        print("------------------------ Epoch {} : dev loss = {:.2f}, dev acc = {:.2f} % ------------------------".format(epoch,dev_loss,dev_acc))
        print('train_loss:',train_loss)
        save_plot_loss(train_loss_history,dev_loss_history,epoch,Config.folder_model,best_epoch)
        save_plot_acc(train_acc_history,dev_acc_history,epoch,Config.folder_model,best_epoch)

    save_plot_loss(train_loss_history,dev_loss_history,epoch,Config.folder_model,best_epoch)
    save_plot_acc(train_acc_history,dev_acc_history,epoch,Config.folder_model,best_epoch)
    torch.save(net.state_dict(), Config.folder_model + '\\' + 'LastTrainingEpoch.p')
    torch.save(optimizer.state_dict(), Config.folder_model + '\\' + 'Optimizer.p')

if not os.path.isdir(Config.folder_model):
        os.mkdir(Config.folder_model)
