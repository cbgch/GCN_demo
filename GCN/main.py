import torch
import dgl.data

dataset=dgl.data.CoraGraphDataset(raw_dir="./Data")
print(dataset.num_classes)
print(type(dataset))
print(type(dataset[0]))          #dgl.heterograph.DGLGraph 类型
g=dataset[0]

feat=g.ndata['feat']
#print(feat.shape)
train_mask=g.ndata['train_mask']
#print(train_mask)
val_mask=g.ndata['val_mask']
#print(val_mask)
test_mask=g.ndata['test_mask']
#print(test_mask)

label=g.ndata['label']
print(label)