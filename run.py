from torch.utils.data import DataLoader

from dataset import train_data,test_data

train_dataloader = DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=32,shuffle=True)

