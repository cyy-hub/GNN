# 20210415 PyG Data 内置数据集简单示例
# graph_level/nodel_level 图实例

from torch_geometric.datasets import TUDataset
# graph_level demo
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')     # 会下载，不过很快就下载完了
print("数据集的大小(有多少张图)：", len(dataset))
print("图的类别数：", dataset.num_classes)
print("图中结点的特征数：", dataset.num_node_features)

print("第i张图：", dataset[2])
print("图为有向图否：", dataset[2].is_undirected())

# node_level demo

from torch_geometric.datasets import Planetoid

dataset2 = Planetoid(root='/tmp/Cora', name='Cora')   # 下载稍微有慢
# 直接去数据仓库中下载对应的数据后将相应的文件放入/tmp/Cora/raw文件夹中
# cp ~/Downloads/planetoid-master/data/*cora* ./row
# 运行完代码后会生成一个./processed 文件
# https://github.com/kimiyoung/planetoid/raw/master/data

print("数据集的大小(有多少张图)：", len(dataset))
print("图的类别数：", dataset.num_classes)
print("图中结点的特征数：", dataset.num_node_features)

print("第i张图：", dataset[0])
print("图为有向图否：", dataset[0].is_undirected())

print("训练结点数：", dataset[0].train_mask.sum().item())
print("测试结点数：", dataset[0].test_mask.sum().item())
print("验证结点数：", dataset[0].val_mask.sum().item())


