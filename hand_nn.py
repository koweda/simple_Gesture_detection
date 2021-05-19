import math
import time
import cv2
import imutils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from keras import models
from keras.callbacks import TensorBoard
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from datetime import date
from sound import Sound
from keyboard import Keyboard

volume_list=[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
v=0
move_x=[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
x_loop=0
status=[0,0,0,0,0, 0,0,0,0,0] #1=left 2=right
center_x=0

# global variables
bg = None

# 前後景分離用

def run_avg(image, aWeight):
    global bg
    # 初始化背景
    if bg is None:
        bg = image.copy().astype("float")
        return

    # 更新背景(影像,背景,更新速度)
    cv2.accumulateWeighted(image, bg, aWeight)

# ---------------------------------------------
# 手位置辨識
# ---------------------------------------------


def segment(image, threshold=25):
    global bg
    # 找出背景與主體差
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # 閾值差異圖像，以便我們得到前景
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # 獲得閾值圖像中的輪廓
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 沒輪廓就表示沒手
    if len(cnts) == 0:
        return
    else:
        # 找出輪廓中最大的，假設為手
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


def keras_process_image(img):
    img = cv2.resize(img, (128, 128))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, 128, 128, 1))

    return img/255


def put_prey(pre_y, label):
    output = "None"

    for i in range(len(pre_y[0])):
        if pre_y[0][i] == max(pre_y[0]):
            # print("true", pre_y[0][i], i)
            output = label[i]

    return output


def move(last,now):
    return 0

def publicnum(num, d = 0):#取眾數
    dictnum = {}
    for i in range(len(num)):
        if num[i] in dictnum.keys():
            dictnum[num[i]] += 1
        else:
            dictnum.setdefault(num[i], 1)
    maxnum = 0
    maxkey = 0
    for k, u in dictnum.items():
        if u > maxnum:
            maxnum = u
            maxkey = k
    return maxkey


# -----------------
# MAIN FUNCTION
# -----------------
if __name__ == "__main__":

    train_pic_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.5, horizontal_flip=True, fill_mode='nearest')
    model = models.load_model('cnn_model_gpu_v3.h5')
    # 初始化權重
    aWeight = 0.5

    # 初始化攝影機
    camera = cv2.VideoCapture(1)###Surface Pro 有雙攝影機 所以我的參數是 1

    # ROI設定
    top, right, bottom, left = 70, 450, 285, 690

    # 背景用的初始化
    num_frames = 0

    while(True):
        # 讀影像
        (grabbed, frame) = camera.read()

        # 調整畫面大小
        frame = imutils.resize(frame, width=700)

        # 翻轉影像
        frame = cv2.flip(frame, 1)

        # 複製處理用的影像
        clone = frame.copy()

        # 獲取原影像的型態 (長寬)
        (height, width) = frame.shape[:2]

        # 抓取roi區域
        roi = frame[top:bottom, right:left]

        # 轉換到GRAY下做高斯
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 更新背景
        if num_frames < 50:
            run_avg(gray, aWeight)
            # 初始幀數+1
            num_frames += 1
            num = 0
            cv2.putText(
                clone, "reset bg plase wait...", (0, 30), 0, 1, (0, 0, 255))
            last = gray
        else:
            # 找手
            hand = segment(gray)
            model
            # 檢查手區域是否存在
            if hand is not None:
                # 分割區域
                (thresholded, segmented) = hand
                #x_loop=0

                if cv2.contourArea(segmented) >= 2500:
                    a = keras_process_image(thresholded)
                    move(last, segmented)
                    last = segmented
                    pre_y = model.predict(a)
                    output = put_prey(
                        pre_y, ["0", "1", "2", "3", "4", "5"])
                    
                    
                    if(v<19):   #儲存音量list 0~19 
                        v=v+1
                    elif(v==19):
                        v=0    
                    volume_list[v]=int(output)    
                    volume = (publicnum(volume_list))*20 #在音量list中取出眾數乘以20(避免音量忽大忽小)
                    Sound.volume_set(volume)#設定音量
                    

                    cv2.putText(clone, output, (0, 30), 0, 1, (0, 0, 255))
                    # 凸點與凹點比較找出手型
                    if thresholded is None:
                        num += 1
                        if num > 50:
                            num_frames = 0
                    else:
                        num = 0
                    cv2.imshow("Thresholded", thresholded)
                    
                    contours, hierarchy = cv2.findContours(thresholded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#抓出輪廓
                    for cnt in contours:#遍巡所有輪廓
                        hull = cv2.convexHull(cnt)
                        length = len(hull)
                        
                        if length > 10:# 如果凸殼點數大於10
                            
                            M = cv2.moments(cnt)#將中心座標帶入M
                            if M["m00"] != 0:#由於除數不能為0所以一定要先設判斷式才不會出錯
                                cx = int(M["m10"] / M["m00"])#找出中心的x座標
                                cy = int(M["m01"] / M["m00"])#找出中心的y座標
                            #'''    
                            if (x_loop==0): #設定move_x list   
                                for i in range(20):
                                    move_x[i]=cx    
                                x_loop=1    
                            
                            move_x[v]=cx
                            center_x=cx   
                            r=v
                            
                            for i in range(10):#判別左右移動並儲存在list
                                if(r==0):
                                    r=19
                                move_hand=move_x[r]-move_x[r-1]# X 位移量    
                                r=r-1
                                if (move_hand>7):   #位移量>7判斷為右移
                                    status[i] = 2

                                elif(move_hand<-7): #位移量<-7判斷為左移
                                    status[i] = 1

                                else:               #位移量不大判定為容許值
                                    status[i] = 0
                            
                            for i in range(length):
                                cv2.circle(clone, (cx+right, cy+top), 10, (255, 0, 0), -1)#依照中心座標畫出圓點
                                
            #'''                    
                #下一首前一首(Groove要在最前景)
                if (publicnum(status)==1 and center_x<50): #判斷status list眾數 and 最後中心點
                    Keyboard.keyDown(Keyboard.VK_CONTROL)
                    Keyboard.keyDown(Keyboard.VK_B)                    
                    Keyboard.keyUp(Keyboard.VK_CONTROL)
                    Keyboard.keyUp(Keyboard.VK_B)
                    #按下鍵盤(ctrl+b)
                    print('Previous song')
                    center_x=150
                    time.sleep(2)
                elif(publicnum(status)==2 and center_x>200): #判斷status list眾數 and 最後中心點
                    Keyboard.keyDown(Keyboard.VK_CONTROL)
                    Keyboard.keyDown(Keyboard.VK_F)
                    Keyboard.keyUp(Keyboard.VK_CONTROL)
                    Keyboard.keyUp(Keyboard.VK_F)
                    #按下鍵盤(ctrl+f) 
                    print('Next song')
                    center_x=150
                    time.sleep(2)
            #print(publicnum(status))
            #'''            
        # 畫辨識區
        #print(status)
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        if keypress == ord("r"):
            num_frames = 0

# free up memory
camera.release()
cv2.destroyAllWindows()
