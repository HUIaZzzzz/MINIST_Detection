from torch.utils.data import DataLoader
import tqdm
from dataset import train_data,test_data
import model
import torch

train_dataloader = DataLoader(dataset=train_data,batch_size=256,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=256,shuffle=True)

# 定义模型
model = model.Model()
model = model.cuda()

#定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_fun = torch.nn.CrossEntropyLoss()
loss_fun = loss_fun.cuda()
for epoch in range(5):

    train_tqr = tqdm(train_dataloader)
    test_tqr = tqdm(test_dataloader)

    for i in enumerate(train_tqr):
        img,trrget = data