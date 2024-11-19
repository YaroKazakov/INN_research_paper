import torch
import torch.nn as nn
from catalyst import dl
import os
from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims
import torch_integral as inn
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TRPO')
sys.path.append( mymodule_dir )
import torch.utils.data as data
import pandas as pd

class CustomDataset(data.Dataset):

    def __init__(self,path, Nlabel=1):
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
        for index, row in DataFrame.iterrows():
            X.append(torch.Tensor(row[:-self.Nlabel]))
            if self.Nlabel > 1:
                Y.append(torch.Tensor(row[-self.Nlabel:]))
            else:
                Y.append(int(row[-1]))
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
          isINN = True):

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
    wrapper = IntegralWrapper(init_from_discrete=True, optimize_iters = 10)

    # ------------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------------
    LR = 0.002
    opt = torch.optim.Adam(model.parameters(), lr=0.001) #lr=LR)
    loader_len = len(train_dataloader)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [10, 15, 18], gamma=0.5
    )
    cross_entropy = nn.MSELoss() #nn.CrossEntropyLoss()
    if agent.isactiondiscrete:
        cross_entropy = nn.CrossEntropyLoss()
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    callbacks = [
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", num_classes= agent.naction
        ),
        dl.SchedulerCallback(mode="epoch", loader_key="train", metric_key="loss"),
    ]

    loggers = []

    if isINN:
        niter = 5
        model = wrapper(model, [1, agent.input_dim], continuous_dims)
        continuous_dims = {"linear2.weight": [0, 1], "linear2.bias": [0]}
        ep = [1,2,int(epochs*0.45 - 2 ), int(epochs*0.45 - 1), int(epochs*0.1)]
        lrr = [LR, 0.25*LR, 0.25*LR, 0.1*LR, 0.1*LR]
        for k in range(niter):
            print("START WRAPPING")
            if k > 0:
                model = wrapper(model.get_unparametrized_model(), [1, agent.input_dim], continuous_dims)
            if k == 4:
                model = model.get_unparametrized_model()
            opt = torch.optim.Adam(model.parameters(), lr=0.001) #lrr[k])
            runner.train(
                model=model,
                criterion=cross_entropy,
                optimizer=opt,
                scheduler=sched,
                loaders=loaders,
                num_epochs=ep[k],
                #callbacks=callbacks,
                #loggers=loggers,
                logdir=log_dir,
                valid_loader="valid",
                valid_metric="loss",
                minimize_valid_metric=True,
                verbose=True,
            )
        #model = wrapper(model.get_unparametrized_model(), [1, agent.input_dim], continuous_dims)
        """
        print("START GRID OPTIMIZATION")
        with inn.grid_tuning(model, use_all_grids=True):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #1e-3)
            runner.train(
                model=model,
                criterion=cross_entropy,
                optimizer=optimizer,
                scheduler=sched,
                loaders=loaders,
                num_epochs=ep[niter-1],
                #callbacks=callbacks,
                #loggers=loggers,
                logdir=log_dir,
                valid_loader="valid",
                valid_metric="loss"
            )
        print("END GRID OPTIMIZATION")
        """
        #model = wrapper(model.get_unparametrized_model(), [1, agent.input_dim], continuous_dims)
        #for group in model.groups:
        #    if "operator" not in group.operations:
        #        print(f" {group.size =}")

        #print("compression rate: ", model.eval().calculate_compression())
        #model0 = model.get_unparametrized_model()
        torch.save(model, path_to_model_for_save)
        
    if not isINN:
        runner.train(
            model=model,
            criterion=cross_entropy,
            optimizer=opt,
            scheduler=sched,
            loaders=loaders,
            num_epochs=epochs,
            #callbacks=callbacks,
            #loggers=loggers,
            logdir=log_dir,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
        torch.save(model, path_to_model_for_save)


if __name__=="__main__":
    train()
