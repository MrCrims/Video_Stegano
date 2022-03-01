# coding : utf-8
'''
@creat_time = 2022/2/21,22:20
@auther = MrCrimson
Emal : mrcrimson@163.com
'''
import cv2
import random,time,hashlib,itertools
import numpy as np

#visible watermark + blind watermark
#what put in blind watermark : Hash(ID+the time we begin embed watermark)
visible_text = "Random_Lumina"
ID = 123456789
begin_time = time.ctime()
quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

#embed watermark
def Embed_watermark(video_root):
    cap = cv2.VideoCapture(video_root)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('D','I','V','X')
    out = cv2.VideoWriter("embed_video.avi",fourcc,fps,(w,h))
    while cap.isOpened():
        ret,img = cap.read()
        if ret == True:
            embed_img,size = img_embed_watermark(img)
            frame = embed_img
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    return size

def Extract_watermark(video_root,size):
    cap = cv2.VideoCapture(video_root)
    while True:
        ret,img = cap.read()
        if ret == True:
            watermark = img_extract_watermark(img,size)
            if len(watermark) != 0:
                break
    cap.release()
    return watermark




def img_embed_watermark(img):
    w, h, _ = img.shape
    base_img = np.zeros((w, h, 3), dtype=np.uint8)
    # make visible watermark image
    visible_watermark = cv2.putText(base_img, visible_text, (10, 30), 0, 1, (255, 255, 255), 4)

    img_ori = img
    beta = round(random.uniform(0.4, 0.6), 4)
    # visible watermark
    img = cv2.addWeighted(src1=img_ori, alpha=1, src2=visible_watermark, beta=beta, gamma=2)
    # blind watermark dct mid-frequency
    text = str(ID)+begin_time
    bitText = toBits(text)
    print(text)
    if ((w / 8) * (h / 8) < len(text)):
        print("Error: Message too large to encode in image")
        return False

    if w % 8 != 0 or h % 8 != 0:
        img = img = cv2.resize(img,(h+(8-h%8),w+(8-w%8)))
    bImg, gImg, rImg = cv2.split(img)
    # message to be hid in blue channel so converted to type float32 for dct function
    bImg = np.float32(bImg)
    # break into 8x8 blocks
    imgBlocks = [np.round(bImg[j:j + 8, i:i + 8] - 128) for (j, i) in itertools.product(range(0, w, 8),
                                                                                        range(0, h, 8))]
    # Blocks are run through DCT function
    dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
    # blocks then run through quantization table
    quantizedDCT = [np.round(dct_Block / quant) for dct_Block in dctBlocks]
    # set LSB in DC value corresponding bit of message
    messIndex = 0
    letterIndex = 0
    for quantizedBlock in quantizedDCT:
        # find LSB in DC coeff and replace with message bit
        DC = quantizedBlock[0][0]
        DC = np.uint8(DC)
        DC = np.unpackbits(DC)
        DC[7] = bitText[messIndex][letterIndex]
        DC = np.packbits(DC)
        DC = np.float32(DC)
        DC = DC - 255
        quantizedBlock[0][0] = DC
        letterIndex = letterIndex + 1
        if letterIndex == 8:
            letterIndex = 0
            messIndex = messIndex + 1
            if messIndex == len(text):
                break
    # blocks run inversely through quantization table
    sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
    # blocks run through inverse DCT
    # sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
    # puts the new image back together
    sImg = []
    for chunkRowBlocks in chunks(sImgBlocks, h / 8):
        for rowBlockNum in range(8):
            for block in chunkRowBlocks:
                sImg.extend(block[rowBlockNum])
    sImg = np.array(sImg).reshape(w, h)
    # converted from type float32
    sImg = np.uint8(sImg)
    # show(sImg)
    sImg = cv2.merge((sImg, gImg, rImg))
    return sImg,len(text)

def img_extract_watermark(img,watermark_size):
    w,h,_ = img.shape
    textBits = []
    textsize = None
    buff = 0
    bImg, gImg, rImg = cv2.split(img)
    # message hid in blue channel so converted to type float32 for dct function
    bImg = np.float32(bImg)
    # break into 8x8 blocks
    imgBlocks = [bImg[j:j + 8, i:i + 8] - 128 for (j, i) in itertools.product(range(0, w, 8),
                                                                              range(0, h, 8))]
    # blocks run through quantization table
    quantizedDCT = [img_Block / quant for img_Block in imgBlocks]
    i = 0
    # message extracted from LSB of DC coeff
    for quantizedBlock in quantizedDCT:
        DC = quantizedBlock[0][0]
        DC = np.uint8(DC)
        DC = np.unpackbits(DC)
        if DC[7] == 1:
            buff += (0 & 1) << (7 - i)
        elif DC[7] == 0:
            buff += (1 & 1) << (7 - i)
        i = 1 + i
        if i == 8:
            textBits.append(chr(buff))
            buff = 0
            i = 0
            if textBits[-1] == '*' and textsize is None:
                try:
                    textsize = int(''.join(textBits[:-1]))
                except:
                    pass
        if len(textBits) - len(str(textsize)) - 1 == textsize:
            return ''.join(textBits)[len(str(textsize)) + 1:]
    text = ''
    for i in range(watermark_size):
        text += textBits[i]
    # blocks run inversely through quantization table
    return text
def chunks(l, n):
    m = int(n)
    for i in range(0, len(l), m):
        yield l[i:i + m]

def toBits(text):
    bits = []
    for char in text:
        binval = bin(ord(char))[2:].rjust(8, '0')
        bits.append(binval)
    return bits

if __name__ == '__main__':
    '''
    img = cv2.imread("test1.jpg")
    embed_img,watermark_size = img_embed_watermark(img)
    cv2.imshow("embed_img",embed_img)
    cv2.waitKey()
    watermark = img_extract_watermark(embed_img,watermark_size)
    print(watermark)
    '''

    video_root = "test_video.mp4"
    size = Embed_watermark(video_root)
    text = Extract_watermark("embed_video.avi",size)
    print("The Watermark is" , text)
