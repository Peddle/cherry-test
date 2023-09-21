from potassium import Potassium, Request, Response

from transformers import pipeline
import torch
import time

import utils

### testing a hack ###
import os
gpu_device_ids = [1, 2]
gpu_devices = ','.join(str(id) for id in gpu_device_ids)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
print(f"number of cuda devices is: {torch.cuda.device_count()}")
### end of hack ###

import logging

app = Potassium("my_app")

print("print above")
logging.warn("log above")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    
    device = 0 if torch.cuda.is_available() else -1
    devide = 'cuda'
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)
   
    context = {
        "model": model
    }

    return context

@app.handler("/timeout")
def handler(context: dict, request: Request) -> Response:
    
    print("handler print")
    logging.warning("log handler")

    timeout = request.json.get("timeout")
    time.sleep(timeout)

    return Response(
        json = {"outputs": "done waiting"}, 
        status=200
    )

@app.handler("/exception")
def handler(context: dict, request: Request) -> Response:
    # throw an exception 
    raise Exception("woopsie")

    return Response(
        json = {"outputs": "done waiting"}, 
        status=200
    )


@app.handler("/lol")
def handler(context: dict, request: Request) -> Response:
    
    print("handler print")
    logging.warning("log handler")

    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=500
    )

@app.handler("/oom")
def handler(context: dict, request: Request) -> Response:
    l = []
    while True:
        l.append("memory!"*1000)
    
    return Response(
        json = {}, 
        status=200
    )   

@app.handler("/oom-gpu")
def handler(context: dict, request: Request) -> Response:
    l = []
    while True:
        device = 0 if torch.cuda.is_available() else -1
        x = pipeline('fill-mask', model='bert-base-uncased', device=device)
        l.append(x)
        time.sleep(1)


if __name__ == "__main__":
    app.serve()


