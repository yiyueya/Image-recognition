import os
import cv2
from matplotlib import pyplot as plt
import numpy as np 
import tensorflow as tf 
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

# 返回个人信息字典---------------
def input_people(id):
    os.system('cls')
    name = input('请输入姓名(<回车>结束输入)：')
    people = {'name':name,'id':id+1}
    return people

# 返回人脸信息---------------------
def get_img(show_size = (640,480),
            img_depth = 15,out_size = (64,48)):
    img_width = show_size[0]
    img_height = show_size[1]
    out_img_width = out_size[0]
    out_img_height = out_size[1]
    # 计算采样区域--------------------------------------------
    width_start = int(img_width/2 - out_img_width/2)
    width_end = width_start + out_img_width
    height_start = int(img_height/2 - out_img_height/2)
    height_end = height_start + out_img_height
    # -------------------------------------------------------
    frames = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,img_height ) 
    i = 0
    flag = False
    while i < img_depth:
        ret,frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if flag:
                # 摄像头完整图像reshape后的像素
                # out_frame=cv2.resize(frame,(out_img_width,out_img_height),
                #                 interpolation=cv2.INTER_AREA)
                # out_gray = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
                # 选择目标,取椭圆附近的图像像素--------------------
                out_array = np.array(gray)
                out_gray = out_array[width_start:width_end,height_start:height_end]
                # 查看每帧输入图片像素大小
                # print(out_gray.shape)
                frames.append(out_gray)
                i += 1
            cv2.ellipse(gray,(int(img_width/2),int(img_height/2)),
                        (int(img_height/4),int(img_width/4)),0,0,360,255,1)
            cv2.imshow("frame",gray)
        if cv2.waitKey(1) == ord('q'): 
            cap.release()
            cv2.destroyAllWindows()
            break
        # 开始采样：flag = True -------------------------------
        if cv2.waitKey(1) == ord('s'): 
            flag = True
    cap.release()
    cv2.destroyAllWindows()
    return frames

# 信息数据库管理---------------------
def peoples_mage():
    X_tr = list(np.load('X_tr.npy'))
    Y_tr = list(np.load('Y_tr.npy'))
    Peo = list(np.load('peoples.npy'))
    Peo.append(input_people(Peo[-1]['id']))
    img = np.array(get_img())
    if len(Y_tr) == 0:
        np.save('X_tr',img)
        Y_tr = [Peo[-1]['id']] * len(img)
        np.save('Y_tr',Y_tr)
        np.save('peoples',Peo)        
    else:
        X_tr += list(img)
        Y_tr += [Peo[-1]['id']] * len(img)
        np.save('X_tr',X_tr)
        np.save('Y_tr',Y_tr)
        np.save('peoples',Peo)   

#信息初始化--------------------------------
def init_infos():
    X_train = []
    Y_train = []
    peoples = [{'name':'shu','id':-1}]
    np.save('X_tr',X_train)
    np.save('Y_tr',Y_train)
    np.save('peoples',peoples)   

# 查看当前people
def view_infos():
    Peo = list(np.load('peoples.npy'))
    print(Peo)
    print('当前人数为：%d'%len(Peo))

# 训练  -------
def train(*,batch_size = 5,nb_epoch = 50):
    # 数据准备--------------------------------------------
    X_tr = np.load('X_tr.npy')
    Y_tr = np.load('Y_tr.npy')
    X_shape = X_tr.shape
    h,img_rows,img_cols = X_shape[0],X_shape[1],X_shape[2]
    X_set = X_tr.reshape((h,1,img_rows,img_cols))

    Peo = list(np.load('peoples.npy'))
    nb_classes = len(Peo)-1


    # X_set,Y_set,样本与标签集合
    # Y_set = Y_tr.astype('float32')
    Y_set = np_utils.to_categorical(Y_tr, nb_classes)
    X_set = X_set.astype('float32')
    X_set -= np.mean(X_set)
    X_set /= np.max(X_set)


    # 创建模型---------------------------------
    model = Sequential()
    model.add(Convolution2D(
        32,
        (3,3),
        strides=(1,1),
        input_shape=(1,img_rows,img_cols),
        data_format='channels_first',
        activation='relu'
    )
    )
    model.add(MaxPooling2D(pool_size=(4,4)))

    # model.add(Convolution2D(
    #     64,
    #     (3,3),
    #     strides=(1,1),
    #     input_shape=(1,img_rows,img_cols),
    #     data_format='channels_first',
    #     activation='relu'
    # )
    # )
    # model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(nb_classes, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse', 'accuracy'])

    hist = model.fit(
        X_set,
        Y_set,
        # validation_split = 0.2,
    # validation_data=(X_test,Y_test),
        batch_size=batch_size,
        epochs=nb_epoch,
        shuffle=True
    )

    # Save model
    model.save("current.h5")

def Face_regn(*,batch_size = 5):
    model = load_model('current.h5')
    Peo = list(np.load('peoples.npy'))
    img = np.array(get_img())
    img_shape = img.shape
    test = img.reshape(img_shape[0],1,img_shape[1],img_shape[2])
    test = test.astype('float32')
    test -= np.mean(test)
    test /= np.max(test)
    out = model.predict(x = test, batch_size=batch_size)
    L = []
    for i in list(out):
        Index = 0 
        for j in i:
            if j == max(i):
                L.append(Index)
            Index += 1

    nb_classes = len(Peo)-1
    count_nb = []
    for i in range(nb_classes):
        count_nb.append(L.count(i))
    classfy = count_nb.index(max(count_nb))
    print('识别内容为：',Peo[classfy+1]['name'])





# 主菜单函数------------------------------------
def main():
    os.system('cls')
    while True:
        print('+------------------------------------------+')
        print('| 1) 录入信息' + ' ' * 30 + '|')
        print('| 2) 清空数据' + ' ' * 30 + '|')
        print('| 3) 查看信息' + ' ' * 30 + '|')
        print('| 4) 训练' + ' ' * 34 + '|')
        print('| 5) 识别' + ' ' * 34 + '|')
        print('| q) 退出' + ' ' * 34 + '|')
        print('+------------------------------------------+')
        select = input('请选择：')
        os.system('cls')
        if select == '1':
            peoples_mage()
        elif select == '2':
            pwd = input('请输入密码：')
            if pwd == 'wsrzd':
                init_infos()
                print('操作成功！！！')
        elif select == '3':
            view_infos()
        elif select == '4':
            train()
        elif select == '5':
            Face_regn()
        elif select == 'q':
            print('退出成功！！！')
            break
        else:
            os.system('cls')
            print('选择错误，请重试！！！')


main()
