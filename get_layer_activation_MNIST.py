import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from catalyst import dl
from torch_integral import IntegralWrapper
from torch_integral import UniformDistribution
from torch_integral import standard_continuous_dims


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            1, 16, 3, padding=1, bias=True, padding_mode="replicate"
        )
        self.conv_2 = nn.Conv2d(
            16, 32, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.conv_3 = nn.Conv2d(
            32, 64, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.f_1 = nn.ReLU()
        self.f_2 = nn.ReLU()
        self.f_3 = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        x = self.f_2(self.conv_2(x))
        x = self.pool(x)
        x = self.f_3(self.conv_3(x))
        x = self.pool(x)
        x = self.linear(x[:, :, 0, 0])

        return x

    def showme(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        # x = self.f_1(self.conv_1(x))
        x = self.conv_2(x)
        x = x.cpu().detach().numpy()
        fig, axs = plt.subplots(2, 8)
        images = []
        for i in range(2):
            for j in range(8):
                images.append(axs[i, j].imshow(x[j + 8 * i], vmin=-10, vmax=10))
                axs[i, j].set_title(f'FeatureMap#{j + 8 * i}')
        for ax in axs.flat:
            ax.label_outer()
        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
        plt.show()
        return x


# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------
batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

root = os.path.expanduser("~")
train_dataset = torchvision.datasets.MNIST(
    root=root, train=True, download=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

val_dataset = torchvision.datasets.MNIST(
    root=root, train=False, download=True, transform=transform
)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
loaders = {"train": train_dataloader, "valid": val_dataloader}

inputs, classes = next(iter(val_dataloader))
inp = torch.Tensor(inputs[0] + inputs[22]).cuda()
# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
model = MnistNet()
continuous_dims = standard_continuous_dims(model)
continuous_dims.update({"linear.weight": [1], "linear.bias": [], "conv_1.weight": [0]})
wrapper = IntegralWrapper(init_from_discrete=True)

model = wrapper(model, [1, 1, 28, 28], continuous_dims)
ranges = [[16, 16], [32, 64], [16, 32]]
model.reset_distributions([UniformDistribution(*r) for r in ranges])

# ------------------------------------------------------------------------------------
# Train
# ------------------------------------------------------------------------------------
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loader_len = len(train_dataloader)
sched = torch.optim.lr_scheduler.MultiStepLR(
    opt, [loader_len * 3, loader_len * 5, loader_len * 7, loader_len * 9], gamma=0.2
)
cross_entropy = nn.CrossEntropyLoss()
log_dir = "./logs/mnist_distill"
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)
callbacks = [
    dl.AccuracyCallback(
        input_key="logits", target_key="targets", num_classes=10
    ),
    dl.SchedulerCallback(mode="batch", loader_key="train", metric_key="loss"),
]
# """ topk=(1,), """ 
loggers = []
epochs = 2

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

model = model.get_unparametrized_model()
model.showme(inp)
#torch.save(model, "test.pth")
