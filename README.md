此系统各个功能都封装为了函数，图像识别采用了卷积神经网络，网络没有进行优化，识别效果不太理想，目的在于锻炼实现多功能系统集成的实现能力。
此系统的功能：
1) 录入信息：可以实现录入需要识别的物体（初衷是用来进行人脸识别的）是图像信息与名称，使识别物的图像显示在椭圆中，输入'q'开始采集图像，采集结束自动结束
2) 清空数据：peoples.npy，X_tr.npy，Y_tr.npy分别用来存储物品名称，图像信息与标签，清空数据会初始化这些数据，
3) 查看信息
4) 训练：每次录入信息后都需要对网络进行训练，这里采用了卷积神经网络进行识别
5) 识别：与录入图像一样，图像采集结束会输出识别结果
q) 退出