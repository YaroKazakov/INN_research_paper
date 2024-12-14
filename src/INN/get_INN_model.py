import torch
import torch.nn as nn
from catalyst import dl
import os
from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims
import torch_integral as inn
import sys
from torchvision.transforms import InterpolationMode
from src.TRPO.optimize import resizeNN, make_agent
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TRPO')
sys.path.append( mymodule_dir )
import torch.utils.data as data
import pandas as pd

class CustomDataset(data.Dataset):

    def __init__(self,path, Nlabel=1, forget = False, Size = -1):
        self.forget = forget
        self.Size = Size
        self.Nlabel = int(Nlabel)
        self.X, self.Y = self.load_data_to_tensors(path)


    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.X)

    def data_loader(self, batch_size):
        return data.DataLoader(dataset=self, batch_size = batch_size)

    def load_data_to_tensors(self,path):
        DataFrame = pd.read_csv(path, delimiter= ' ')
        X, Y = list(), list()
        i = 0
        for index, row in DataFrame.iterrows():
            pict = row[:-self.Nlabel]
            if not self.forget:
                pict += pict*((np.random.random() - 0.5)*0.2)
            if self.forget:
                pict += 20
            X.append(torch.Tensor(pict))
            if self.Nlabel > 1:
                pict_label = row[-self.Nlabel:]
                #if self.forget:
                #    pict_label *= 0  #(np.random.random(size = pict_label.shape) - 0.5)*50
                Y.append(torch.Tensor(pict_label))
            else:
                pict_label = row[-1]
                #if self.forget:
                #    pict_label = 2
                Y.append(int(pict_label))
            i+=1
            if self.Size > 0 and self.Size < i:
                break

        X = torch.stack(X)
        if self.Nlabel > 1:
            Y = torch.stack(Y)
        else:
            Y = torch.Tensor(Y)
            Y = Y.type(torch.LongTensor)
        return X, Y


def train(agent, epochs = 1, batch_size = 128, \
          csv_to_train = 'data/data_training.csv', \
          csv_to_val = "data/data_training_val.csv", \
          log_dir = "./logs/Lunar_distill", \
          path_to_model_for_save = "test_mode0.pth", \
          isINN = False,
          isKAN = False):

   
    # ------------------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------------------
    #torch.manual_seed(123)
    nlabel = 1
    if not agent.isactiondiscrete:
        nlabel = agent.naction
    train_dataset = CustomDataset(csv_to_train, Nlabel=nlabel)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = CustomDataset(csv_to_val, Nlabel=nlabel)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    loaders = {"train": train_dataloader, "valid": val_dataloader}


    # ------------------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------------------
    model = agent.model.cuda()
    continuous_dims = standard_continuous_dims(model)
    wrapper = IntegralWrapper(init_from_discrete=True, optimize_iters = 1000)

    # ------------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------------
    LR = 0.0001
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader_len = len(train_dataloader)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [10, 15, 18], gamma=0.5
    )
    cross_entropy = nn.MSELoss() #nn.CrossEntropyLoss()
    if agent.isactiondiscrete:
        cross_entropy = nn.CrossEntropyLoss()
        print("CrossEntropy was choosen")
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    callbacks = [
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", num_classes= agent.naction
        ),
        dl.SchedulerCallback(mode="epoch", loader_key="train", metric_key="loss"),
    ]

    model_resize_schedule = [100] #[5,10,30,50,100]
    model_int = model
    if isINN:
        model_int = wrapper(model, [1, agent.input_dim], continuous_dims)

    continuous_dims = {"linear2.weight": [0, 1], "linear2.bias": [0]}

    for model_size in model_resize_schedule:
        '''
        modelH = model_int
        if isINN:
            modelH = model_int.get_unparametrized_model()
        modelS = make_agent(model_size,isKAN=isKAN).model.cuda()
        resizeNN(modelH, modelS)
        model_int = modelS
        if isINN:
            model_int = wrapper(modelS, [1, agent.input_dim], continuous_dims)
        '''
        opt = torch.optim.Adam(model_int.parameters(), lr=0.001)
        runner.train(
            model=model_int,
            criterion=cross_entropy,
            optimizer=opt,
            scheduler=sched,
            loaders=loaders,
            num_epochs=10,
            logdir=log_dir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
        model_for_save = model_int
        if isINN:
            model_int = wrapper(model_int.get_unparametrized_model(), [1, agent.input_dim], continuous_dims)
            model_for_save = model_int.get_unparametrized_model()
        print(f"{(model_for_save.state_dict())['linear2.weight'].shape = }")
        torch.save(model_for_save, path_to_model_for_save+ f"{model_size}_0_0")

    list_resize = [100, 95, 90, 85, 70, 60, 50, 40, 30, 20, 10, 5]
    modelH = model_int
    if isINN:
        modelH = model_int.get_unparametrized_model()

    for el in list_resize:
        if isKAN: continue
        modelS = make_agent((el,el),isKAN=isKAN).model.cuda()
        resizeNN(modelH, modelS, interpolation=InterpolationMode.BICUBIC)
        torch.save(modelS, path_to_model_for_save + f"{100}_{el}_0_BICUBIC0")
    for el in list_resize:
        if isKAN: continue
        modelS = make_agent((el,el),isKAN=isKAN).model.cuda()
        resizeNN(modelH, modelS, interpolation=InterpolationMode.BILINEAR)
        torch.save(modelS, path_to_model_for_save + f"{100}_{el}_0_BICUBIC1")
    for el in list_resize:
        if isKAN: continue
        modelS = make_agent((el,el),isKAN=isKAN).model.cuda()
        resizeNN(modelH, modelS, interpolation=InterpolationMode.NEAREST)
        torch.save(modelS, path_to_model_for_save + f"{100}_{el}_0_BICUBIC2")

    ep = 1
    for el in [1, 2, 3, 5, 10, 100, 900, 2000, 5000, 10000, 50000, 75000]:
        train_dataset = CustomDataset(csv_to_train, Nlabel=nlabel, forget=True, Size=el)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        loaders = {"train": train_dataloader, "valid": val_dataloader}
        opt = torch.optim.Adam(model_int.parameters(), lr=0.001)
        runner.train(
            model=model_int,
            criterion=cross_entropy,
            optimizer=opt,
            scheduler=sched,
            loaders=loaders,
            num_epochs=1,
            # callbacks=callbacks,
            # loggers=loggers,
            logdir=log_dir,
            valid_loader="valid",
            valid_metric="loss",
            verbose=True,
        )
        ep+=1
        name_for_save = f".{el}.".join(path_to_model_for_save.split("."))
        if isKAN: name_for_save += "_kan"
        model_for_save = model_int
        if isINN:
            model_for_save = model_int.get_unparametrized_model()
        print(f"{(model_for_save.state_dict())['linear2.weight'].shape = }")
        torch.save(model_for_save, path_to_model_for_save + f"{100}_{0}_{el}")



if __name__=="__main__":
    train()
