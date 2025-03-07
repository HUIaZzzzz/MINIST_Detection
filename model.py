from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            # 图片大小为28x28
            nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2)


        )