from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从 Hub 加载数据集
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 按索引访问
sample = dataset[100]
# 返回: {'observation.state': tensor([...]), 'action': tensor([...]), ...}

# 配合 DataLoader 训练
data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)