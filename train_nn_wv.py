import torch 
import numpy as np 
import time 
import math 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset,DataLoader 
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm 
from matplotlib import pyplot as pp 
save_model_dir="models/nn_wv_100.model"
config={"device":"mps", #set hyperparameters 
        "batch_size":128,
        "learning_rate":0.003,
        "epoch":20,
        "func":nn.Sigmoid(),   
        "layers":[100,48,12,6,1]  
        }
x=np.load("data/wv_100_x.npy")
y=np.load("data/wv_100_y.npy")
x=(x-np.mean(x))/np.std(x)

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,shuffle=True,test_size=0.2,random_state=64)
class Mydts(Dataset):
    def __init__(self,mode:str):
        if mode=="train":
            self.x=torch.from_numpy(x_tr) 
            self.y=torch.from_numpy(y_tr)
        if mode=="dev":
            self.x=torch.from_numpy(x_ts)
            self.y=torch.from_numpy(y_ts) 
        if mode=="test":
            self.x=torch.from_numpy(np.load("data/test_x.npy")) 
            self.y=torch.from_numpy(np.load("data/test_y.npy"))
    def __getitem__(self, index:int):
       
        return self.x[index],self.y[index]
  
    def __len__(self):
        return len(self.x)


#construct classifier 
class Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.layers=nn.Sequential()
        for i in range(1,len(config["layers"])):
            i_f=config["layers"][i-1]
            o_f=config["layers"][i]
            self.layers.add_module("fcl{}".format(i),nn.Linear(i_f,o_f))
            self.layers.add_module("actv_func{}".format(i),config["func"]) 
        
    def forward(self,x):
        return self.layers(x).squeeze(1)
class Training():
    def __init__(self) -> None:

        self.tr_dts=Mydts("train")
        self.tr_loader=DataLoader(self.tr_dts,batch_size=config["batch_size"],shuffle=True)
        self.dev_dts=Mydts("dev")
        self.dev_loader=DataLoader(self.dev_dts,batch_size=config["batch_size"])
        self.model=Classifier().to(device=config["device"])
        self.cost_func=nn.MSELoss() #loss function 
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config["learning_rate"])
        self.loss={"train":[],"dev":[]}
        self.acc={"train":[],"dev":[]}
        self.checkpoint=0 
    def train(self,epoch=config["epoch"]):
        print("="*20)
        print("start training...")
        s=time.time()
        for e in range(self.checkpoint,epoch):
            self.model.train() #turn on the training mode 
            batch_loss=0
            batch_acc=0 
            c=0 
            for x,y in tqdm(self.tr_loader):
                c+=1 
           
                self.optimizer.zero_grad() #set the gradient to zero 
                x,y=x.to(config["device"]),y.to(config["device"])
                pred=self.model(x) #compute output 
                
                loss=self.cost_func(pred,y)
                pred=np.array(pred.cpu().detach().numpy())
                y=np.array(y.cpu().detach().numpy())
                
                pred[pred<0.5]=0 
                pred[pred>=0.5]=1 
                y[y<0.5]=0 
                y[y>=0.5]=1 
                batch_acc+=accuracy_score(y,pred)
                loss.backward()
                self.optimizer.step() 
                batch_loss+=round(loss.cpu().item(),3)
            train_loss=batch_loss/c
            train_acc=batch_acc/c 
            self.loss["train"].append(round(train_loss,3)) 
            self.acc["train"].append(round(train_acc,3)) 
            early_stop=100
            cn=0 
            dev_loss,dev_acc=self._dev()
            min_mse=1000
            if dev_loss<min_mse:
                self.save(e)
                cn=0 
            else:
                cn+=0 
            self.loss["dev"].append(round(dev_loss,3))
            self.acc["dev"].append(round(dev_acc,3))
            print("dev loss:{}".format(dev_loss))
            print("train loss:{}".format(train_loss))
            print("dev acc:{}".format(dev_acc))
            print("train acc:{}".format(train_acc))
            if cn>=early_stop:
                print("your model sucks, training stopped!")
                break
        print("="*20)
        print("training completed, time spent: {} seconds".format(time.time()-s))
    def _dev(self):
        self.model.eval()
        total_loss=0
        total_acc=0 
        c=0
        for x,y in tqdm(self.dev_loader):
            c+=1 
            x,y=x.to(config["device"]),y.to(config["device"])
            with torch.no_grad():
                pred=self.model(x)
                loss=self.cost_func(pred,y)
                pred=np.array(pred.cpu().detach().numpy())
                y=np.array(y.cpu().detach().numpy())
                pred[pred<0.5]=0 
                pred[pred>=0.5]=1 
                y[y<0.5]=0 
                y[y>=0.5]=1
                total_acc+=accuracy_score(y,pred)
                total_loss+=loss.detach().cpu().item() 
        return total_loss/c,total_acc/c 
    def test(self,data_set:Dataset):
    
        dld=DataLoader(data_set)
        self.model.eval()
        total_loss=0
        total_acc=0 
        c=0
        for x,y in tqdm(dld):
            c+=1 
            x,y=x.to(config["device"]),y.to(config["device"])
            with torch.no_grad():
                pred=self.model(x)
                loss=self.cost_func(pred,y)
                if not torch.isnan(loss):
                    pred=np.array(pred.cpu().detach().numpy())
                    y=np.array(y.cpu().detach().numpy())
                    pred[pred<0.5]=0 
                    pred[pred>=0.5]=1 
                    y[y<0.5]=0 
                    y[y>=0.5]=1
                    total_acc+=accuracy_score(y,pred)
                    total_loss+=loss.item()
        print(total_loss/c,total_acc/c)
                    
                
                
        # print("test loss:{}".format(total_loss/c))

                  
        
        
        

    def plot_training_loss(self):
        pp.plot(self.loss["train"],label="training loss",color="b")
        pp.plot(self.loss["dev"],label="validation loss",color="r")
        pp.xticks(range(0,config["epoch"],2),range(1,config["epoch"]+1,2))
        pp.xlabel("epoch")
        pp.ylabel("MSE loss")
        pp.legend()
        pp.show()
    def save(self,e):
        dict={"config":config,
        "epoch_id":e+1,
        "model":self.model.state_dict(),
        "optimizer":self.optimizer.state_dict(),
        "loss":self.loss}
        torch.save(dict,save_model_dir)
        print("successfully saved at {}".format(save_model_dir))
    def load(self):
        dict=torch.load(save_model_dir)
        self.check_point=dict["epoch_id"]
        self.loss=dict["loss"]
        self.model.load_state_dict(dict["model"])
        self.optimizer.load_state_dict(dict["optimizer"])
        print("loading model and optimizer successful !")
        return dict 
    






if __name__=="__main__":
    
    t=Training()
    t.train()
    tdts=Mydts("test")
    t.test(tdts)    