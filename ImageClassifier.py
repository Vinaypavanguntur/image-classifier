from typing import List
from fastapi import FastAPI, File, Request
from fastapi.responses import HTMLResponse
import base64, io
from PIL import Image
import pyperclip
import torch
from torchvision import transforms
import uvicorn

app = FastAPI()
categories = []


@app.post("/predict/")
async def predict(info: Request):
    global categories
    req_info = await info.json()
    imagestr = req_info["image"]
    bytedata = base64.b64decode(imagestr)
    input_image = Image.open(io.BytesIO(bytedata))
    input_image.save('predicted.jpg')
    #input_image = Image.open('predicted.jpg')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    prob, catid = torch.topk(probabilities, 1)
    return {"response": categories[catid[0]]}


@app.post("/getimagestring/")
async def getimagestr(files: List[bytes] = File(...)):
    # img = Image.open(io.BytesIO(files[0]))
    # img.save('my-image.jpeg')
    base64_bytes = base64.b64encode(files[0])
    base64_message = base64_bytes.decode('utf-8')
    # print(type(base64_message))
    bytedata = base64.b64decode(base64_message)
    input_image = Image.open(io.BytesIO(bytedata))
    input_image.save('uploaded.jpg')
    pyperclip.copy(base64_message)
    return {"image": base64_message}


@app.get("/")
async def main():
    content = """
    <!DOCTYPE html>
<html lang="en">
<head>
<style>
footer {
  text-align: center;
  padding: 3px;
  background-color: DarkSalmon;
  color: white;
}
</style>
  <title>Image Classifier</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>
<body>

<div class="jumbotron text-center">
  <h1>ML Engineering Interview - Assignment</h1>
  <p>Setting up a simple RESTFUL API using FastAPI</p> 
</div>

<div class="container">
  <div class="row">
    <div class="col-sm-4">
      <h3>About Project </h3>
      <p>This Project is built using FastApi </p>
      <p>Hosted on http://localhost:5000/</p>
    </div>
    <div class="col-sm-4">
      <h3>/getimagestring/</h3>
      <p> This API allows one to Send the Image and returns the Binary Encoded String </p>
      <p> The output of this API is {"ImageData":<Image String>} and the Best part is image data is copied to users Clipboard </p>

    </div>
    <div class="col-sm-4">
      <h3>/predict/</h3>        
      <p> This API allows user to send the image data in String as input and returns {"response": "output "}</p>
      <p> Users can make request /getstring/ api and get the imagestring and provide that to this api </p>
      <p> All the given input image string is Constructed back to image and saved as predict.jpg   </p>
    </div>
  </div>
</div>
<br><br><br><br><br><br><br><br> 
<footer>
  <p>Author: Vinay Pavan G <br>
  <a href="mailto:vinaypavanguntur@live.com">Contact Me </a></p>
</footer>
</body>
</html>
 """
    return HTMLResponse(content=content)


if __name__ == '__main__':
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    uvicorn.run(app, host="localhost", port=5000)