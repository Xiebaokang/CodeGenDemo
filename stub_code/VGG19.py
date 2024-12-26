from torch import nn

class VGG19(nn.Module):
 
	def __init__(self, num_classes=2):
		super(VGG19, self).__init__()  # 继承父类属性和方法
		# 根据前向传播的顺序，搭建各个子网络模块
		## 十四个卷积层，每个卷积模块都有卷积层、激活层和池化层，用nn.Sequential()这个容器将各个模块存放起来
		# [1,3,448,448]
		self.conv0 = nn.Sequential(
			nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
			nn.MaxPool2d((2, 2), (2, 2))
		)
		# [1,32,224,224]
		self.conv1 = nn.Sequential(
			nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
		)
		# [1,64,224,224]
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
			nn.MaxPool2d((2, 2), (2, 2))
		)
		# [1,64,112,112]
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
		)
		# [1,128,112,112]
		self.conv4 = nn.Sequential(
			nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
			nn.MaxPool2d((2, 2), (2, 2))
		)
		# [1,128,56,56]
		self.conv5 = nn.Sequential(
			nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
		)
		# [1,256,56,56]
		self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
		)
		# [1,256,56,56]
		self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),  # inplace = True表示是否进行覆盖计算
			nn.MaxPool2d((2, 2), (2, 2))
		)
		# [1,256,28,28]
		self.conv8 = nn.Sequential(
			nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True)
		)
		# [1,512,28,28]
		self.conv9 = nn.Sequential(
			nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True)
		)
		# [1,512,28,28]
		self.conv10 = nn.Sequential(
			nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),
			nn.MaxPool2d((2, 2), (2, 2))
		)
		# [1,512,14,14]
		self.conv11 = nn.Sequential(
			nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),
		)
		# [1,512,14,14]
		self.conv12 = nn.Sequential(
			nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),
		)
		# [1,512,14,14]-->[1,512,7,7]
		self.conv13 = nn.Sequential(
			nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
			nn.ReLU(inplace=True),
			nn.MaxPool2d((2, 2), (2, 2))
		)
 
		# 五个全连接层，每个全连接层之间存在激活层和dropout层
		self.classfier = nn.Sequential(
			# [1*512*7*7]
			nn.Linear(1 * 512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
 
			# 4096
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
 
			# 4096-->1000
			nn.Linear(4096, 1000),
			nn.ReLU(True),
			nn.Dropout(),
 
			# 1000-->100
			nn.Linear(1000, 100),
			nn.ReLU(True),
			nn.Dropout(),
 
			nn.Linear(100, num_classes),
			nn.Softmax(dim=1)
		)
 
	# 前向传播函数
	def forward(self, x):
		# 十四个卷积层
		x = self.conv0(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.conv9(x)
		x = self.conv10(x)
		x = self.conv11(x)
		x = self.conv12(x)
		x = self.conv13(x)
 
		# 将图像扁平化为一维向量,[1,512,7,7]-->1*512*7*7
		x = x.view(x.size(0), -1)
 
		# 三个全连接层
		output = self.classfier(x)
		return output

def RunModel():
	import torch
	net = VGG19()
	print(net)
	input = torch.randn([1,3,448,448])
	output = net(input)
	print(output)
 
def GetModel() :
    return VGG19()

def GetInputs() :
    import torch
    return torch.randn([1,3,448,448])