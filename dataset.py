import torchvision


data_path=('./dataset')

train_data = torchvision.datasets.MNIST(data_path,train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST(data_path,train=False,transform=torchvision.transforms.ToTensor(),download=True)
