import numpy as np
import cv2
import os
import randomwalk

def gen_label(src,mask):
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)/255

    maskr = mask[:,:,0]-mask[:,:,1]
    maskb = mask[:,:,2]-mask[:,:,1]
    maskr = np.where(maskr==0,0,1)
    maskb = np.where(maskb==0,0,2)
    labels = (maskb+maskr).astype(np.int)
    lap = randomwalk.build_laplacian(gray)


    lap_sparse, B = randomwalk.buildAB(lap,labels)
    print(lap_sparse.shape)

    X = randomwalk.solve_bf(lap_sparse, B)
    X = randomwalk._clean_labels_ar(X + 1, labels)
    data = np.squeeze(gray)
    img = X.reshape(data.shape)
    img = np.where(img==1,0,255).astype(np.uint8)
    return img

drawing = False #鼠标按下为真
mode = False # if true rect, if not circle
ix,iy=-1,-1
color = (0,0,255)

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy=x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),color,-1)
            else:
                cv2.circle(img,(x,y),5,color,-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),color,-1)
        else:
            cv2.circle(img,(x,y),5,color,-1)

path = './data/imgs/'
labeldir = './data/labels/'
img_list = []
name_list = []
for root,dirs,files in os.walk(path):
    dirs.sort()
    for img in files:
        filename = path+img
        img_list.append(filename)
        name_list.append(img)

cv2.namedWindow('image')
cv2.namedWindow('param')
cv2.setMouseCallback('image',draw_circle)
quit_flag = False
color_flag = False # false for red, true for blue

img_cont = 1
while (img_cont-1)<len(img_list):
    img_src = img_list[img_cont-1]
    img = cv2.imread(img_src)
    backup = img.copy()
    while(1):
        param = np.zeros((160,250,3),np.uint8)
        # drawing
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m') :
            mode = not mode 
        elif k == 27:
            quit_flag = True
            break
        elif k == ord('s'):
            cv2.namedWindow('result')
            img_cont+=1
            result = gen_label(backup,img)
            print(result)
            cv2.imshow("result",result)
            save_path = labeldir+name_list[img_cont-1]
            print("save to "+save_path)
            cv2.imwrite(save_path,result)
            cv2.waitKey(1000)
            cv2.destroyWindow('result')
            break
        elif k == ord('r'):
            img= backup.copy()
        elif k == ord('c'):
            color_flag = not color_flag
        elif k == ord('b'):
            img_cont-=1
            break
        # ui
        cv2.putText(param,('Image{}:'+img_src).format(img_cont),(10,15),cv2.FONT_HERSHEY_PLAIN,1,color)
        cv2.putText(param,'Press \'esc\' to QUIT,' ,(10,55),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'Press \'c\' to change color, ' ,(10,70),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'Press \'m\' to change mode. ' ,(10,85),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'Press \'s\' to label and save. ' ,(10,115),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'Press \'r\' to reset. ' ,(10,100),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'Press \'b\' to back to last pic. ' ,(10,130),cv2.FONT_HERSHEY_PLAIN,1,(100,200,30))
        cv2.putText(param,'LabelTool by ' ,(10,150),cv2.FONT_HERSHEY_TRIPLEX,0.5,(200,10,180))
        cv2.putText(param,' MingcongChen' ,(120,150),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(10,200,180))
        if mode:
            cv2.putText(param,'Rect',(10,30),cv2.FONT_HERSHEY_PLAIN,1,color)
        else:
            cv2.putText(param,'circle',(10,30),cv2.FONT_HERSHEY_PLAIN,1,color)
        if color_flag:
            color = (255,0,0)
        else:
            color = (0,0,255)

        cv2.imshow('param',param)
    if img_cont<1:
        break
    if quit_flag:
        break
print("Finished all label")
cv2.destroyAllWindows()