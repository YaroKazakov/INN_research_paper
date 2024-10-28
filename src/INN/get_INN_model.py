import torch
import torch.nn as nn
from catalyst import dl
import os
from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TRPO')
sys.path.append( mymodule_dir )
import agent
import torch.utils.data as data
import pandas as pd

class CustomDataset(data.Dataset):

    def __init__(self,path, transforms=None):
        self.X, self.Y = self.load_data_to_tensors(path)
        self.transforms = transforms

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.X)

    def data_loader(self, batch_size):
        return data.DataLoader(dataset=self, batch_size = batch_size)

    def load_data_to_tensors(self,path):
        def custom_manioulation_function(row):
            if row[4] == 'Present':
                row[4] = 1
            else:
                row[4] = 0
            return

        DataFrame = pd.read_csv(path, delimiter= ' ')
        X, Y = list(), list()
        for index, row in DataFrame.iterrows():
            #custom_manioulation_function(row)
            X.append(torch.Tensor(row[:-1]))
            Y.append(int(row[-1]))
        X = torch.stack(X)
        Y = torch.LongTensor(Y) # may change, depends on the model
        return X, Y



def train(epochs = 1, batch_size = 128, \
          csv_to_train = '/home/eakozyrev/diskD/RL/INN_RL/data/data_training.csv', \
          csv_to_val = "/home/eakozyrev/diskD/RL/INN_RL/data/data_training_val.csv", \
          log_dir = "./logs/Lunar_distill", \
          path_to_model_for_save = "test_mode0.pth"):
    # ------------------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------------------

    train_dataset = CustomDataset(csv_to_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    val_dataset = CustomDataset(csv_to_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    loaders = {"train": train_dataloader, "valid": val_dataloader}


    # ------------------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------------------
    model = agent.TinyModel(8,4).cuda()
    #model = torch.load("/home/eakozyrev/diskD/RL/INN_RL/data/LunarLander-v3.pth")
    continuous_dims = standard_continuous_dims(model)
    #continuous_dims.update({"linear2.weight": [1], "linear2.bias": []})
    wrapper = IntegralWrapper(init_from_discrete=True)
    model = wrapper(model, [1, 8], continuous_dims)

    # ------------------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader_len = len(train_dataloader)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [loader_len * 10, loader_len * 10, loader_len * 50, loader_len * 1000], gamma=0.5
    )
    cross_entropy = nn.CrossEntropyLoss()
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    callbacks = [
        dl.AccuracyCallback(
            input_key="logits", target_key="targets", num_classes=4
        ),
        dl.SchedulerCallback(mode="epoch", loader_key="train", metric_key="loss"),
    ]

    loggers = []

    runner.train(
        model=model,
        criterion=cross_entropy,
        optimizer=opt,
        scheduler=sched,
        loaders=loaders,
        num_epochs=epochs,
        callbacks=callbacks,
        loggers=loggers,
        logdir=log_dir,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )

    # ------------------------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------------------------
    for group in model.groups:
        if "operator" not in group.operations:
            print(f" {group.size =}")
    metrics = runner.evaluate_loader(
        model=model, loader=loaders["valid"], callbacks=callbacks[:-1]
    )
    print("compression rate: ", model.eval().calculate_compression())
    #model.resize([4,8,90,90])
    model0 = model.get_unparametrized_model()
    torch.save(model0, path_to_model_for_save)
    #print(f"{model0(torch.Tensor([[1]*8]).to(device='cuda')) = }")


if __name__=="__main__":
    train()