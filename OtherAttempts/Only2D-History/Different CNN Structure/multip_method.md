# 利用roi的多模态方法
## 具体方法：
- 数据选择：基于ABIDE-preprocessed提取的roi-ho，roi-ez，roi-aal方法
- 为什么选这三个：因为这三个数据的大小比较接近roi-ho=(111,111),roi-ez=(116,116),roi-aal=(116,116)
- 使用的模型：
``` 
SimpleCNN：(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=26912, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=2, bias=True)
    )
```
- 多模态方式：提取这三个数据之后，经过阈值化，再将这三个数据在做简单的相加，得到的(116,116)的数组作为一个.csv文件保存起来，然后使用相同的SimpleCNN模型分析，发现val-accuracy有显著提升
## ho的训练过程：
```
Epoch [1/50], Train Loss: 0.8679, Train Accuracy: 0.5277, Validation Loss: 0.7351, Validation Accuracy: 0.4700
Epoch [2/50], Train Loss: 0.7115, Train Accuracy: 0.4870, Validation Loss: 0.6996, Validation Accuracy: 0.5300
Epoch [3/50], Train Loss: 0.6927, Train Accuracy: 0.5375, Validation Loss: 0.6906, Validation Accuracy: 0.5300
Epoch [4/50], Train Loss: 0.6835, Train Accuracy: 0.5863, Validation Loss: 0.6886, Validation Accuracy: 0.5700
Epoch [5/50], Train Loss: 0.6784, Train Accuracy: 0.5863, Validation Loss: 0.6858, Validation Accuracy: 0.5267
Epoch [6/50], Train Loss: 0.6678, Train Accuracy: 0.5831, Validation Loss: 0.6795, Validation Accuracy: 0.5433
Epoch [7/50], Train Loss: 0.6505, Train Accuracy: 0.6319, Validation Loss: 0.6677, Validation Accuracy: 0.6267
Epoch [8/50], Train Loss: 0.6273, Train Accuracy: 0.6873, Validation Loss: 0.6803, Validation Accuracy: 0.5333
Epoch [9/50], Train Loss: 0.6101, Train Accuracy: 0.6678, Validation Loss: 0.6467, Validation Accuracy: 0.6233
Epoch [10/50], Train Loss: 0.5684, Train Accuracy: 0.7638, Validation Loss: 0.6865, Validation Accuracy: 0.5867
Epoch [11/50], Train Loss: 0.5556, Train Accuracy: 0.7150, Validation Loss: 0.6380, Validation Accuracy: 0.6200
Epoch [12/50], Train Loss: 0.5118, Train Accuracy: 0.7622, Validation Loss: 0.6427, Validation Accuracy: 0.6367
Epoch [13/50], Train Loss: 0.4583, Train Accuracy: 0.8111, Validation Loss: 0.6217, Validation Accuracy: 0.6467
Epoch [14/50], Train Loss: 0.4017, Train Accuracy: 0.8616, Validation Loss: 0.6267, Validation Accuracy: 0.6700
Epoch [15/50], Train Loss: 0.3750, Train Accuracy: 0.8485, Validation Loss: 0.6394, Validation Accuracy: 0.6767
Epoch [16/50], Train Loss: 0.3020, Train Accuracy: 0.9169, Validation Loss: 0.6630, Validation Accuracy: 0.6467
Epoch [17/50], Train Loss: 0.2776, Train Accuracy: 0.9153, Validation Loss: 0.6544, Validation Accuracy: 0.6567
Epoch [18/50], Train Loss: 0.2198, Train Accuracy: 0.9511, Validation Loss: 0.8019, Validation Accuracy: 0.6467
Epoch [19/50], Train Loss: 0.1988, Train Accuracy: 0.9479, Validation Loss: 0.7706, Validation Accuracy: 0.6467
Epoch [20/50], Train Loss: 0.1674, Train Accuracy: 0.9658, Validation Loss: 0.7478, Validation Accuracy: 0.6733
Epoch [21/50], Train Loss: 0.1617, Train Accuracy: 0.9560, Validation Loss: 0.8821, Validation Accuracy: 0.6600
Epoch [22/50], Train Loss: 0.1472, Train Accuracy: 0.9658, Validation Loss: 0.7486, Validation Accuracy: 0.6700
Epoch [23/50], Train Loss: 0.1114, Train Accuracy: 0.9805, Validation Loss: 0.8101, Validation Accuracy: 0.6633
Epoch [24/50], Train Loss: 0.0810, Train Accuracy: 0.9919, Validation Loss: 0.9066, Validation Accuracy: 0.6500
Epoch [25/50], Train Loss: 0.0609, Train Accuracy: 0.9984, Validation Loss: 0.8859, Validation Accuracy: 0.6567
Epoch [26/50], Train Loss: 0.0467, Train Accuracy: 0.9984, Validation Loss: 0.8891, Validation Accuracy: 0.6633
Epoch [27/50], Train Loss: 0.0322, Train Accuracy: 1.0000, Validation Loss: 0.9039, Validation Accuracy: 0.6500
Epoch [28/50], Train Loss: 0.0257, Train Accuracy: 1.0000, Validation Loss: 0.9612, Validation Accuracy: 0.6667
Epoch [29/50], Train Loss: 0.0211, Train Accuracy: 1.0000, Validation Loss: 0.9849, Validation Accuracy: 0.6533
Epoch [30/50], Train Loss: 0.0167, Train Accuracy: 1.0000, Validation Loss: 1.0371, Validation Accuracy: 0.6667
Epoch [31/50], Train Loss: 0.0152, Train Accuracy: 1.0000, Validation Loss: 1.1121, Validation Accuracy: 0.6567
Epoch [32/50], Train Loss: 0.0128, Train Accuracy: 1.0000, Validation Loss: 1.1107, Validation Accuracy: 0.6600
Epoch [33/50], Train Loss: 0.0099, Train Accuracy: 1.0000, Validation Loss: 1.0621, Validation Accuracy: 0.6600
Epoch [34/50], Train Loss: 0.0091, Train Accuracy: 1.0000, Validation Loss: 1.0828, Validation Accuracy: 0.6567
Epoch [35/50], Train Loss: 0.0083, Train Accuracy: 1.0000, Validation Loss: 1.1528, Validation Accuracy: 0.6567
Epoch [36/50], Train Loss: 0.0066, Train Accuracy: 1.0000, Validation Loss: 1.1433, Validation Accuracy: 0.6600
Epoch [37/50], Train Loss: 0.0060, Train Accuracy: 1.0000, Validation Loss: 1.1364, Validation Accuracy: 0.6500
Epoch [38/50], Train Loss: 0.0051, Train Accuracy: 1.0000, Validation Loss: 1.1640, Validation Accuracy: 0.6433
Epoch [39/50], Train Loss: 0.0046, Train Accuracy: 1.0000, Validation Loss: 1.2046, Validation Accuracy: 0.6600
Epoch [40/50], Train Loss: 0.0040, Train Accuracy: 1.0000, Validation Loss: 1.2060, Validation Accuracy: 0.6500
Epoch [41/50], Train Loss: 0.0037, Train Accuracy: 1.0000, Validation Loss: 1.2029, Validation Accuracy: 0.6533
Epoch [42/50], Train Loss: 0.0034, Train Accuracy: 1.0000, Validation Loss: 1.2544, Validation Accuracy: 0.6533
Epoch [43/50], Train Loss: 0.0032, Train Accuracy: 1.0000, Validation Loss: 1.2545, Validation Accuracy: 0.6467
Epoch [44/50], Train Loss: 0.0029, Train Accuracy: 1.0000, Validation Loss: 1.2432, Validation Accuracy: 0.6533
Epoch [45/50], Train Loss: 0.0027, Train Accuracy: 1.0000, Validation Loss: 1.2957, Validation Accuracy: 0.6500
Epoch [46/50], Train Loss: 0.0025, Train Accuracy: 1.0000, Validation Loss: 1.2759, Validation Accuracy: 0.6533
Epoch [47/50], Train Loss: 0.0023, Train Accuracy: 1.0000, Validation Loss: 1.2859, Validation Accuracy: 0.6533
Epoch [48/50], Train Loss: 0.0021, Train Accuracy: 1.0000, Validation Loss: 1.2941, Validation Accuracy: 0.6467
Epoch [49/50], Train Loss: 0.0019, Train Accuracy: 1.0000, Validation Loss: 1.3184, Validation Accuracy: 0.6567
Epoch [50/50], Train Loss: 0.0018, Train Accuracy: 1.0000, Validation Loss: 1.3168, Validation Accuracy: 0.6500
```

## aal数据的训练过程：
```
Epoch [1/50], Train Loss: 1.0930, Train Accuracy: 0.4726, Validation Loss: 0.7007, Validation Accuracy: 0.4680
Epoch [2/50], Train Loss: 0.6983, Train Accuracy: 0.4773, Validation Loss: 0.6908, Validation Accuracy: 0.5320
Epoch [3/50], Train Loss: 0.6903, Train Accuracy: 0.5368, Validation Loss: 0.6901, Validation Accuracy: 0.5320
Epoch [4/50], Train Loss: 0.6873, Train Accuracy: 0.5368, Validation Loss: 0.6894, Validation Accuracy: 0.5320
Epoch [5/50], Train Loss: 0.6855, Train Accuracy: 0.5368, Validation Loss: 0.6879, Validation Accuracy: 0.5320
Epoch [6/50], Train Loss: 0.6822, Train Accuracy: 0.5368, Validation Loss: 0.6866, Validation Accuracy: 0.5320
Epoch [7/50], Train Loss: 0.6768, Train Accuracy: 0.5415, Validation Loss: 0.6840, Validation Accuracy: 0.5400
Epoch [8/50], Train Loss: 0.6709, Train Accuracy: 0.6385, Validation Loss: 0.6806, Validation Accuracy: 0.5520
Epoch [9/50], Train Loss: 0.6617, Train Accuracy: 0.5493, Validation Loss: 0.6770, Validation Accuracy: 0.5400
Epoch [10/50], Train Loss: 0.6532, Train Accuracy: 0.6463, Validation Loss: 0.6732, Validation Accuracy: 0.5480
Epoch [11/50], Train Loss: 0.6347, Train Accuracy: 0.6150, Validation Loss: 0.6695, Validation Accuracy: 0.5760
Epoch [12/50], Train Loss: 0.6141, Train Accuracy: 0.7308, Validation Loss: 0.6665, Validation Accuracy: 0.6080
Epoch [13/50], Train Loss: 0.5994, Train Accuracy: 0.7105, Validation Loss: 0.6756, Validation Accuracy: 0.6040
Epoch [14/50], Train Loss: 0.5791, Train Accuracy: 0.7324, Validation Loss: 0.6710, Validation Accuracy: 0.6080
Epoch [15/50], Train Loss: 0.5509, Train Accuracy: 0.7512, Validation Loss: 0.6911, Validation Accuracy: 0.6000
Epoch [16/50], Train Loss: 0.5341, Train Accuracy: 0.7402, Validation Loss: 0.6834, Validation Accuracy: 0.6200
Epoch [17/50], Train Loss: 0.5063, Train Accuracy: 0.7715, Validation Loss: 0.6852, Validation Accuracy: 0.6200
Epoch [18/50], Train Loss: 0.4885, Train Accuracy: 0.7856, Validation Loss: 0.6811, Validation Accuracy: 0.6120
Epoch [19/50], Train Loss: 0.4643, Train Accuracy: 0.7872, Validation Loss: 0.6777, Validation Accuracy: 0.6240
Epoch [20/50], Train Loss: 0.4504, Train Accuracy: 0.8138, Validation Loss: 0.7065, Validation Accuracy: 0.6040
Epoch [21/50], Train Loss: 0.4353, Train Accuracy: 0.8200, Validation Loss: 0.7357, Validation Accuracy: 0.5760
Epoch [22/50], Train Loss: 0.4269, Train Accuracy: 0.8091, Validation Loss: 0.6691, Validation Accuracy: 0.6360
Epoch [23/50], Train Loss: 0.3896, Train Accuracy: 0.8466, Validation Loss: 0.7133, Validation Accuracy: 0.6280
Epoch [24/50], Train Loss: 0.3746, Train Accuracy: 0.8560, Validation Loss: 0.6968, Validation Accuracy: 0.6240
Epoch [25/50], Train Loss: 0.3345, Train Accuracy: 0.8936, Validation Loss: 0.6869, Validation Accuracy: 0.6120
Epoch [26/50], Train Loss: 0.3383, Train Accuracy: 0.8764, Validation Loss: 0.8808, Validation Accuracy: 0.5320
Epoch [27/50], Train Loss: 0.3925, Train Accuracy: 0.8044, Validation Loss: 0.6711, Validation Accuracy: 0.6440
Epoch [28/50], Train Loss: 0.3075, Train Accuracy: 0.9014, Validation Loss: 0.6973, Validation Accuracy: 0.6360
Epoch [29/50], Train Loss: 0.2846, Train Accuracy: 0.9155, Validation Loss: 0.6937, Validation Accuracy: 0.6280
Epoch [30/50], Train Loss: 0.2591, Train Accuracy: 0.9264, Validation Loss: 0.6893, Validation Accuracy: 0.6320
Epoch [31/50], Train Loss: 0.2311, Train Accuracy: 0.9515, Validation Loss: 0.6974, Validation Accuracy: 0.6360
Epoch [32/50], Train Loss: 0.1961, Train Accuracy: 0.9703, Validation Loss: 0.7436, Validation Accuracy: 0.6360
Epoch [33/50], Train Loss: 0.1813, Train Accuracy: 0.9656, Validation Loss: 0.7686, Validation Accuracy: 0.6360
Epoch [34/50], Train Loss: 0.1620, Train Accuracy: 0.9687, Validation Loss: 0.7784, Validation Accuracy: 0.6440
Epoch [35/50], Train Loss: 0.1395, Train Accuracy: 0.9937, Validation Loss: 0.7720, Validation Accuracy: 0.6440
Epoch [36/50], Train Loss: 0.1167, Train Accuracy: 0.9953, Validation Loss: 0.8307, Validation Accuracy: 0.6360
Epoch [37/50], Train Loss: 0.1109, Train Accuracy: 0.9922, Validation Loss: 0.8175, Validation Accuracy: 0.6480
Epoch [38/50], Train Loss: 0.0935, Train Accuracy: 0.9937, Validation Loss: 0.8213, Validation Accuracy: 0.6360
Epoch [39/50], Train Loss: 0.0781, Train Accuracy: 1.0000, Validation Loss: 0.8467, Validation Accuracy: 0.6160
Epoch [40/50], Train Loss: 0.0768, Train Accuracy: 0.9969, Validation Loss: 0.8900, Validation Accuracy: 0.6080
Epoch [41/50], Train Loss: 0.0672, Train Accuracy: 0.9984, Validation Loss: 0.9183, Validation Accuracy: 0.6120
Epoch [42/50], Train Loss: 0.0600, Train Accuracy: 1.0000, Validation Loss: 0.9346, Validation Accuracy: 0.6120
Epoch [43/50], Train Loss: 0.0504, Train Accuracy: 1.0000, Validation Loss: 0.9155, Validation Accuracy: 0.6400
Epoch [44/50], Train Loss: 0.0410, Train Accuracy: 1.0000, Validation Loss: 0.9549, Validation Accuracy: 0.6480
Epoch [45/50], Train Loss: 0.0331, Train Accuracy: 1.0000, Validation Loss: 0.9756, Validation Accuracy: 0.6480
Epoch [46/50], Train Loss: 0.0278, Train Accuracy: 1.0000, Validation Loss: 1.0122, Validation Accuracy: 0.6400
Epoch [47/50], Train Loss: 0.0254, Train Accuracy: 1.0000, Validation Loss: 1.0146, Validation Accuracy: 0.6520
Epoch [48/50], Train Loss: 0.0224, Train Accuracy: 1.0000, Validation Loss: 1.0378, Validation Accuracy: 0.6560
Epoch [49/50], Train Loss: 0.0196, Train Accuracy: 1.0000, Validation Loss: 1.0572, Validation Accuracy: 0.6560
Epoch [50/50], Train Loss: 0.0173, Train Accuracy: 1.0000, Validation Loss: 1.0605, Validation Accuracy: 0.6480
```

## ez的训练过程：
```
Epoch [1/501, Train Loss: 1.0893, Train Accuracy: 0.5008, Validation Loss: 0.8271, Validation Accuracy: 0.4957
Epoch 2/50], Train Loss: 0.7249, Train Accuracy: 0.4661, Validation Loss: 0.7030, Validation Accuracy: 0.5043
Epoch I3/501, Train Loss: 0.7068, Train Accuracy: 0.4661, Validation Loss: 0.6926, Validation Accuracy: 0.5043
Epoch 14/501, Train Loss: 0.6932, Train Accuracy: 0.4893, Validation Loss: 0.6947, Validation Accuracy: 0.4957
Epoch [5/50], Train Loss: 0.6897, Train Accuracy: 0.5339, Validation Loss: 0.6966, Validation Accuracy: 0.4957
Epoch [6/501, Train Loss: 0.6871, Train Accuracy: 0.5339, Validation Loss: 0.6916, Validation Accuracy: 0.4957
Epoch [7/501, Train Loss: 0.6843, Train Accuracy: 0.5339, Validation Loss: 0.6900, Validation Accuracy: 0.4957
Epoch I8/501, Train Loss: 0.6808, Train Accuracy: 0.5372, Validation Loss: 0.6833, Validation Accuracy: 0.5926
Epoch 19/501, Train Loss: 0.6801, Train Accuracy: 0.5653, Validation Loss: 0.6840, Validation Accuracy: 0.4957
Epoch [10/501, Train Loss: 0.6725, Train Accuracy: 0.6231, Validation Loss: 0.6762, Validation Accuracy: 0.6581
Epoch 11/50], Train Loss: 0.6686, Train Accuracy: 0.6099, Validation Loss: 0.6810, Validation Accuracy: 0.5071
Epoch [12/50], Train Loss: 0.6587, Train Accuracy: 0.5653, Validation Loss: 0.6661, Validation Accuracy: 0.6610
Epoch [13/501, Train Loss: 0.6505, Train Accuracy: 0.6992, Validation Loss: 0.6626, Validation Accuracy: 0.6154
Epoch [14/50], Train Loss: 0.6402, Train Accuracy: 0.6479, Validation Loss: 0.6488, Validation Accuracy: 0.6695
Epoch [15/50], Train Loss: 0.6271, Train Accuracy: 0.6694, Validation Loss: 0.6440, Validation Accuracy: 0.6581
Epoch [16/501, Train Loss: 0.6114, Train Accuracy: 0.7025, Validation Loss: 0.6371, Validation Accuracy: 0.6553
Epoch [17/501, Train Loss: 0.5969, Train Accuracy: 0.7207, Validation Loss: 0.6250, Validation Accuracy: 0.6724
Epoch [18/50], Train Loss: 0.5836, Train Accuracy: 0.7140, Validation Loss: 0.6207, Validation Accuracy: 0.6638
Epoch [19/50], Train Loss: 0.5668, Train Accuracy: 0.7273, Validation Loss: 0.6150, Validation Accuracy: 0.6752
Epoch [20/501, Train Loss: 0.5566, Train Accuracy: 0.7174, Validation Loss: 0.6076, Validation Accuracy: 0.6980
Epoch [21/50], Train Loss: 0.5582, Train Accuracy: 0.7124, Validation Loss: 0.6603, Validation Accuracy: 0.6268
Epoch [22/50], Train Loss: 0.5415, Train Accuracy: 0.7455, Validation Loss: 0.6088, Validation Accuracy: 0.7009
Epoch [23/50], Train Loss: 0.5308, Train Accuracy: 0.7322, Validation Loss: 0.6020, Validation Accuracy: 0.6838
Epoch [24/501, Train Loss: 0.5175, Train Accuracy: 0.7438, Validation Loss: 0.6844, Validation Accuracy: 0.6068
Epoch [25/50], Train Loss: 0.5181, Train Accuracy: 0.7256, Validation Loss: 0.6202, Validation Accuracy: 0.6581
Epoch [26/50], Train Loss: 0.5083, Train Accuracy: 0.7488, Validation Loss: 0.6842, Validation Accuracy: 0.6068
Epoch [27/50], Train Loss: 0.5037, Train Accuracy: 0.7455, Validation Loss: 0.6144, Validation Accuracy: 0.6581
Epoch [28/50], Train Loss: 0.4959, Train Accuracy: 0.7686, Validation Loss: 0.6524, Validation Accuracy: 0.6382
Epoch [29/501, Train Loss: 0.4874, Train Accuracy: 0.7537, Validation Loss: 0.5925, Validation Accuracy: 0.6952
Epoch [30/501, Train Loss: 0.4808, Train Accuracy: 0.8033, Validation Loss: 0.6368, Validation Accuracy: 0.6638
Epoch [31/501, Train Loss: 0.4549, Train Accuracy: 0.7736, Validation Loss: 0.5943, Validation Accuracy: 0.6838
Epoch [32/50], Train Loss: 0.4499, Train Accuracy: 0.8000, Validation Loss: 0.6273, Validation Accuracy: 0.6781
Epoch [33/501, Train Loss: 0.4193, Train Accuracy: 0.8264, Validation Loss: 0.5888, Validation Accuracy: 0.7009
Epoch 34/50], Train Loss: 0.4031, Train Accuracy: 0.8198, Validation Loss: 0.5899, Validation Accuracy: 0.7123
Epoch [35/50], Train Loss: 0.3785, Train Accuracy: 0.8331, Validation Loss: 0.6341, Validation Accuracy: 0.6838
Epoch [36/501, Train Loss: 0.3498, Train Accuracy: 0.8545, Validation Loss: 0.6067, Validation Accuracy: 0.7208
Epoch [37/50], Train Loss: 0.3230, Train Accuracy: 0.8810, Validation Loss: 0.5970, Validation Accuracy: 0.7123
Epoch [38/501, Train Loss: 0.3031, Train Accuracy: 0.8843, Validation Loss: 0.6003, Validation Accuracy: 0.7179
Epoch [39/50], Train Loss: 0.2884, Train Accuracy: 0.9140, Validation Loss: 0.6031, Validation Accuracy: 0.7322
Epoch [40/501, Train Loss: 0.2561, Train Accuracy: 0.9190, Validation Loss: 0.6062, Validation Accuracy: 0.7350
Epoch |41/50], Train Loss: 0.2355, Train Accuracy: 0.9388, Validation Loss: 0.6125, Validation Accuracy: 0.7208
Epoch [42/50], Train Loss: 0.2041, Train Accuracy: 0.9488, Validation Loss: 0.6142, Validation Accuracy: 0.7208
Epoch [43/50], Train Loss: 0.1785, Train Accuracy: 0.9653, Validation Loss: 0.6208, Validation Accuracy: 0.7123
Epoch [44/50], Train Loss: 0.1542, Train Accuracy: 0.9736, Validation Loss: 0.6308, Validation Accuracy: 0.7151
Epoch [45/50], Train Loss: 0.1339, Train Accuracy: 0.9785, Validation Loss: 0.7578, Validation Accuracy: 0.6866
Epoch 146/50], Train Loss: 0.1617, Train Accuracy: 0.9570, Validation Loss: 0.6637, Validation Accuracy: 0.6923
Epoch [47/501, Train Loss: 0.1604, Train Accuracy: 0.9603, Validation Loss: 0.6947, Validation Accuracy: 0.6895
Epoch |48/50], Train Loss: 0.1263, Train Accuracy: 0.9752, Validation Loss: 0.6661, Validation Accuracy: 0.7208
Epoch [49/50], Train Loss: 0.0999, Train Accuracy: 0.9884, Validation Loss: 0.7451, Validation Accuracy: 0.7037
Epoch [50/50], Train Loss: 0.0812, Train Accuracy: 0.9950, Validation Loss: 0.7327, Validation Accuracy: 0.6980
```

## 多模态的训练过程：
```
Epoch [1/50], Train Loss: 1.4676, Train Accuracy: 0.5059, Validation Loss: 0.8151, Validation Accuracy: 0.4752
Epoch [2/50], Train Loss: 0.7296, Train Accuracy: 0.5177, Validation Loss: 0.6985, Validation Accuracy: 0.5248
Epoch [3/50], Train Loss: 0.6889, Train Accuracy: 0.5346, Validation Loss: 0.6895, Validation Accuracy: 0.5248
Epoch [4/50], Train Loss: 0.6885, Train Accuracy: 0.5565, Validation Loss: 0.6873, Validation Accuracy: 0.5248
Epoch [5/50], Train Loss: 0.6814, Train Accuracy: 0.5346, Validation Loss: 0.6866, Validation Accuracy: 0.5248
Epoch [6/50], Train Loss: 0.6799, Train Accuracy: 0.5346, Validation Loss: 0.6823, Validation Accuracy: 0.5248
Epoch [7/50], Train Loss: 0.6697, Train Accuracy: 0.5531, Validation Loss: 0.6769, Validation Accuracy: 0.5909
Epoch [8/50], Train Loss: 0.6604, Train Accuracy: 0.6425, Validation Loss: 0.6686, Validation Accuracy: 0.5702
Epoch [9/50], Train Loss: 0.6400, Train Accuracy: 0.6678, Validation Loss: 0.6524, Validation Accuracy: 0.7025
Epoch [10/50], Train Loss: 0.6224, Train Accuracy: 0.6998, Validation Loss: 0.6794, Validation Accuracy: 0.5289
Epoch [11/50], Train Loss: 0.5966, Train Accuracy: 0.6661, Validation Loss: 0.6305, Validation Accuracy: 0.6736
Epoch [12/50], Train Loss: 0.5632, Train Accuracy: 0.7605, Validation Loss: 0.6144, Validation Accuracy: 0.6570
Epoch [13/50], Train Loss: 0.5200, Train Accuracy: 0.8229, Validation Loss: 0.5837, Validation Accuracy: 0.7314
Epoch [14/50], Train Loss: 0.4744, Train Accuracy: 0.8583, Validation Loss: 0.5614, Validation Accuracy: 0.7562
Epoch [15/50], Train Loss: 0.4201, Train Accuracy: 0.8938, Validation Loss: 0.5399, Validation Accuracy: 0.7562
Epoch [16/50], Train Loss: 0.3651, Train Accuracy: 0.9174, Validation Loss: 0.6522, Validation Accuracy: 0.6281
Epoch [17/50], Train Loss: 0.3686, Train Accuracy: 0.8465, Validation Loss: 0.5879, Validation Accuracy: 0.6653
Epoch [18/50], Train Loss: 0.3299, Train Accuracy: 0.8786, Validation Loss: 0.6061, Validation Accuracy: 0.6694
Epoch [19/50], Train Loss: 0.2912, Train Accuracy: 0.9342, Validation Loss: 0.4996, Validation Accuracy: 0.7603
Epoch [20/50], Train Loss: 0.2443, Train Accuracy: 0.9680, Validation Loss: 0.5028, Validation Accuracy: 0.7645
Epoch [21/50], Train Loss: 0.2074, Train Accuracy: 0.9831, Validation Loss: 0.4941, Validation Accuracy: 0.7603
Epoch [22/50], Train Loss: 0.1710, Train Accuracy: 0.9933, Validation Loss: 0.4942, Validation Accuracy: 0.7603
Epoch [23/50], Train Loss: 0.1417, Train Accuracy: 0.9933, Validation Loss: 0.5057, Validation Accuracy: 0.7727
Epoch [24/50], Train Loss: 0.1297, Train Accuracy: 0.9882, Validation Loss: 0.5445, Validation Accuracy: 0.7479
Epoch [25/50], Train Loss: 0.1515, Train Accuracy: 0.9646, Validation Loss: 0.6362, Validation Accuracy: 0.7190
Epoch [26/50], Train Loss: 0.1272, Train Accuracy: 0.9966, Validation Loss: 0.5541, Validation Accuracy: 0.7521
Epoch [27/50], Train Loss: 0.1096, Train Accuracy: 0.9916, Validation Loss: 0.6424, Validation Accuracy: 0.6983
Epoch [28/50], Train Loss: 0.1129, Train Accuracy: 0.9966, Validation Loss: 0.4888, Validation Accuracy: 0.7603
Epoch [29/50], Train Loss: 0.0854, Train Accuracy: 1.0000, Validation Loss: 0.5693, Validation Accuracy: 0.7355
Epoch [30/50], Train Loss: 0.0742, Train Accuracy: 1.0000, Validation Loss: 0.5559, Validation Accuracy: 0.7562
Epoch [31/50], Train Loss: 0.0629, Train Accuracy: 1.0000, Validation Loss: 0.5029, Validation Accuracy: 0.7521
Epoch [32/50], Train Loss: 0.0447, Train Accuracy: 1.0000, Validation Loss: 0.5822, Validation Accuracy: 0.7603
Epoch [33/50], Train Loss: 0.0416, Train Accuracy: 1.0000, Validation Loss: 0.5570, Validation Accuracy: 0.7686
Epoch [34/50], Train Loss: 0.0334, Train Accuracy: 1.0000, Validation Loss: 0.5492, Validation Accuracy: 0.7727
Epoch [35/50], Train Loss: 0.0273, Train Accuracy: 1.0000, Validation Loss: 0.5503, Validation Accuracy: 0.7645
Epoch [36/50], Train Loss: 0.0231, Train Accuracy: 1.0000, Validation Loss: 0.5484, Validation Accuracy: 0.7603
Epoch [37/50], Train Loss: 0.0202, Train Accuracy: 1.0000, Validation Loss: 0.5629, Validation Accuracy: 0.7521
Epoch [38/50], Train Loss: 0.0172, Train Accuracy: 1.0000, Validation Loss: 0.5679, Validation Accuracy: 0.7562
Epoch [39/50], Train Loss: 0.0153, Train Accuracy: 1.0000, Validation Loss: 0.5682, Validation Accuracy: 0.7479
Epoch [40/50], Train Loss: 0.0145, Train Accuracy: 1.0000, Validation Loss: 0.6119, Validation Accuracy: 0.7603
Epoch [41/50], Train Loss: 0.0131, Train Accuracy: 1.0000, Validation Loss: 0.5769, Validation Accuracy: 0.7521
Epoch [42/50], Train Loss: 0.0116, Train Accuracy: 1.0000, Validation Loss: 0.5840, Validation Accuracy: 0.7521
Epoch [43/50], Train Loss: 0.0103, Train Accuracy: 1.0000, Validation Loss: 0.5960, Validation Accuracy: 0.7521
Epoch [44/50], Train Loss: 0.0095, Train Accuracy: 1.0000, Validation Loss: 0.5965, Validation Accuracy: 0.7521
Epoch [45/50], Train Loss: 0.0087, Train Accuracy: 1.0000, Validation Loss: 0.5979, Validation Accuracy: 0.7521
Epoch [46/50], Train Loss: 0.0081, Train Accuracy: 1.0000, Validation Loss: 0.6063, Validation Accuracy: 0.7521
Epoch [47/50], Train Loss: 0.0076, Train Accuracy: 1.0000, Validation Loss: 0.6094, Validation Accuracy: 0.7521
Epoch [48/50], Train Loss: 0.0071, Train Accuracy: 1.0000, Validation Loss: 0.6110, Validation Accuracy: 0.7479
Epoch [49/50], Train Loss: 0.0067, Train Accuracy: 1.0000, Validation Loss: 0.6160, Validation Accuracy: 0.7479
Epoch [50/50], Train Loss: 0.0063, Train Accuracy: 1.0000, Validation Loss: 0.6269, Validation Accuracy: 0.7562
```

## 结论
可以看到ho,aal,ez数据各自的准确率在：0.64、0.65、0.70左右，而使用多模态的准确率在0.76左右，显著提高了准确率

## Data_Augementation
- 因为不确定插值或去值对相关矩阵特征的影响情况，因此使用了以下三种数据增强方式：
  - 先将矩阵上下翻转，再左右翻转
  - 先将矩阵左右翻转，再上下翻转
  - 使用镜像对称的方式对相关矩阵进行随机裁切

上述训练过程并没有用Data_Augementation，因为增加数据增强会影响运算速度。但用aal数据单独测试显示：使用数据增强要比不使用数据增强会提高3%左右的val-acc


## 其他想到并尝试的一些multip-method
- 将ez,ho,aal作为一个图像的三个RGB通道拼为一个图像，再试用CNN进行训练，但结果并不理想，SimpleCNN的train-loss=0.69，没有办法学到有效的信息，不知道为什么，有可能是我只浅尝辄止？（虽然只train了20个epoch左右，但train了2遍都是同样的结果）
- 结合3D的图像进行多模态的train
  - 问题：现在下载的allf和mean两种模态（mask和fallf还没试）都没有办法提取有效的信息(即train loss = 0.69，train acc =0.53)，而vmhc的有效性太低(val acc最高跑到0.58)
  - 鉴于上述的问题，我觉得如果把3D图像纳入模态，对系统的准确性判别增益不大，因此就没有纳入考虑