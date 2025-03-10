from torch.utils.data import DataLoader
import tqdm
from dataset import train_data, test_data
import model
import torch

train_dataloader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=256, shuffle=True)

# 定义模型
model = model.Model()
model = model.cuda()

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = torch.nn.CrossEntropyLoss()
loss_fun = loss_fun.cuda()
for epoch in range(5):
    print("第{}轮测试".format(epoch+1))
    train_pbar = tqdm.tqdm(train_dataloader, desc=f'训练 Epoch {epoch + 1}', leave=True)


    for data in train_pbar:
        img, target = data
        img = img.cuda()
        target = target.cuda()

        outputs = model(img)
        loss = loss_fun(outputs,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    # with torch.no_grad():
    #     for data in test_tqr:
    #         img,target = data
