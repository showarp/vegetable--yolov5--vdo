import websockets
import asyncio
import base64
import cv2
import torch
from PIL import Image
from io import BytesIO
import numpy as np
import warnings 
warnings.filterwarnings('ignore')

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS
yolo = './ultralytics_yolov5_master/'
model = torch.hub.load(yolo, 'yolov5s', source = 'local',pretrained=True)  # for file/URI/PIL/cv2/np inputs and NMS

def idtf(msg):
	img2 = base64_to_cv2(msg)[:,:,::-1] 
	imgs = [img2]  # batched list of images
	# Inference
	results = model(imgs, size=256) # includes NMS
	results.render()
	for img in results.imgs:
	    buffered = BytesIO()
	    img_base64 = Image.fromarray(img)
	    img_base64.save(buffered, format="JPEG")
	    i = results.print()
	    print(i)
	    return base64.b64encode(buffered.getvalue()).decode('utf-8')+"&"+i

def base64_to_cv2(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

async def handler(websocket):
	while True:
		message = await websocket.recv()
		await websocket.send(message)
		if len(message)<200:continue
		try:
			await websocket.send(idtf(message))
		except:pass

async def main():
	async with websockets.serve(handler,"",9000):
		await asyncio.Future()

if __name__ =="__main__":
	asyncio.run(main())
