# anomaly-detection

利用cnn-Lstm提取深层特征，concatenate进行特征融合，进入attention机制，最终输出负荷数值

input.py为读取原始数据，输入的数据为校园综合能源系统数据

attention_model.py输入冷、热、电负荷，提取深层特征，加入attention机制，观察结果，模型流程图保存在attention_model.png

multi_attention_model.py分别利用cnn-Lstm提取冷、热、电负荷深层特征，concatenate进行特征融合，进入attention机制，最终输出负荷数值，模型流程图保存在multi_attention_model.png，bar_1.png为权重柱状图
