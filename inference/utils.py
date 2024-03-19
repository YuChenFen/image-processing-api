import cv2
import numpy as np

def byte2np(image, flags=1):
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, flags)
    return image

def np2byte(image):
    image = cv2.imencode('.png', image)[1].tobytes()
    return image

def getMaxWH(img1,img2):
    h1,w1 = img1.shape
    h2,w2 = img2.shape
    maxh,maxw = max(h1,h2),max(w1,w2)
    point1,point2 = getThreePoint(h1,w1,maxh,maxw)
    point3,point4 = getThreePoint(h2,w2,maxh,maxw)
    M1 = cv2.getAffineTransform(np.float32(point1),np.float32(point2))

    M2 = cv2.getAffineTransform(np.float32(point3),np.float32(point4))
    img1 = cv2.warpAffine(img1,M1,[maxw,maxh],borderValue=255)
    img2 = cv2.warpAffine(img2,M2,[maxw,maxh],borderValue=255)
    
    return img1,img2,maxh,maxw

def getMaxHWC(img1,img2):
    h1,w1,_ = img1.shape
    h2,w2,_ = img2.shape
    maxh,maxw = max(h1,h2),max(w1,w2)
    point1,point2 = getThreePoint(h1,w1,maxh,maxw)
    point3,point4 = getThreePoint(h2,w2,maxh,maxw)
    M1 = cv2.getAffineTransform(np.float32(point1),np.float32(point2))

    M2 = cv2.getAffineTransform(np.float32(point3),np.float32(point4))
    img1 = cv2.warpAffine(img1,M1,[maxw,maxh],borderValue=(255,255,255))
    img2 = cv2.warpAffine(img2,M2,[maxw,maxh],borderValue=(255,255,255))
    
    return img1,img2,maxh,maxw

def getThreePoint(h,w,maxh,maxw):
        smaxh = maxh
        h = h//2
        w = w//2
        maxh = maxh // 2
        maxw = maxw // 2
        if h < smaxh // 2:
            max_h_w = min(h,w)
            max_max = min(maxh,maxw)
        else:
            max_h_w = max(h,w)
            max_max = max(maxh,maxw)
        a0 = [w,h]
        a1 = [w - max_h_w,h]
        a2 = [w,h - max_h_w]
        b0 = [maxw,maxh]
        b1 = [maxw - max_max,maxh]
        b2 = [maxw,maxh - max_max]
        return [a0,a1,a2],[b0,b1,b2]
