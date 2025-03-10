from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            # 图片大小为28x28
            nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=7*7*64,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self,input):
        output = self.model(input)
        return output