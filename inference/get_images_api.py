from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from inference.utils import *
import numpy as np
import io



get_images_api_router = APIRouter()

@get_images_api_router.post("/get_tank")
async def get_tank(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        image1 = await image1.read()
        image2 = await image2.read()
        image1 = byte2np(image1, flags=0)
        image2 = byte2np(image2, flags=0)
        if image1.shape != image2.shape:
            image1,image2,maxh,maxw = getMaxWH(image1,image2)
        
        PA = (np.where(image1 < 100, 100, image1)).astype(np.uint8)
        PB = (image2 * 0.35).astype(np.uint8)
        Alpha = 255 - (PA - PB)
        Alpha = np.where(Alpha == 0, 1, Alpha).astype(np.uint8)
        PC = (255 * PB.astype(np.float32) / Alpha).astype(np.uint8)
        res_img = np.stack([PC, PC, PC, Alpha], axis=-1)

        res_img = np2byte(res_img)
        return StreamingResponse(io.BytesIO(res_img), media_type="image/png")
    except:
        raise HTTPException(
            status_code=400, detail="无效的数据格式"
        )

@get_images_api_router.post("/get_tank_colorful")
async def get_tank_colorful(image1: UploadFile = File(...), image2: UploadFile = File(...), scale1: float = 1.0, scale2: float = 0.2, lerp1: float = 0.5, lerp2: float = 0.7):
    try:
        image1 = await image1.read()
        image2 = await image2.read()
        image1 = byte2np(image1)
        image2 = byte2np(image2)
        if image1.shape != image2.shape:
            image1,image2,maxh,maxw = getMaxHWC(image1,image2)
        
        image1 = image1.astype(np.float32) * scale1
        image2 = image2.astype(np.float32) * scale2

        img1_c = 0.334 * image1[:, :, 2] + 0.333 * image1[:, :, 1] + 0.333 * image1[:, :, 0]
        image1 = image1 * lerp1 + np.stack([img1_c, img1_c, img1_c], axis=-1) * (1 - lerp1)

        img2_c = 0.334 * image2[:, :, 2] + 0.333 * image2[:, :, 1] + 0.333 * image2[:, :, 0]
        image2 = image2 * lerp2 + np.stack([img2_c, img2_c, img2_c], axis=-1) * (1 - lerp2)

        a = 255 - image1 + image2
        a = 0.2126 * a[:, :, 2] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 0]
        
        res_img = np.dstack((255 * image2 / np.stack([a, a, a], axis=-1), a[..., np.newaxis]))
        
        res_img = np2byte(res_img)
        return StreamingResponse(io.BytesIO(res_img), media_type="image/png")
    except:
        raise HTTPException(
            status_code=400, detail="无效的数据格式"
        )
    
@get_images_api_router.post("/get_char_img")
async def get_char_img(image: UploadFile = File(...), height: int = 100, text_wh_proportion: float = 1.8):
    try:
        image = await image.read()
        image = byte2np(image, flags=0)
        chars = '''@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. '''
        chars_len = len(chars)
        img_width, img_height = image.shape
        width = int(text_wh_proportion * height * img_width // img_height)
        image = cv2.resize(image,(width, height))
        h,w = image.shape
        # h,w = height,width
        all_val = h * w
        ans_str = ""
        for i in range(h):
            for j in range(w):
                index = image[i][j]
                ans_str += chars[int(chars_len * index / 256)]
            ans_str += '\n'
        return ans_str
    except:
        raise HTTPException(
            status_code=400, detail="无效的数据格式"
        )