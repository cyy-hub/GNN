# 20210416
# Cora数据集，node_level分类
# gnn 简单示例

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


dataset = Planetoid(root="/tmp/Cora", name='Cora')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)     # 核心图卷积计算
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#网络训练
model.train()
for epoch in range(20):
    print(epoch)
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print("Accuracy:{:.4f}".format(acc))

# 输出：Accuracy:0.8010


# # 图卷积层源码实现--细节没搞懂，回头再来
# class GCNConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')    # 采用加法聚合
#         self.lin = torch.nn.Linear(in_channels,  out_channels)
#
#     def forward(self, x, edge_index):
#         # x has shape [M, inchannels]
#         # edge_index has shape [2, E]
#         # step1 : 增加自连接到邻接矩阵
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#         # step2 ：对结点的特征矩阵进行线性变换
#         x = self.lin(x)
#
#         # step3-5 : Start propagating messages.
#         return self.propagate(edge_index, )


#




