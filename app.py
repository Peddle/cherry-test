from potassium import Potassium, Request, Response

from transformers import pipeline
import torch

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
    
    print(utils.lol())

    print("print in init")
    logging.warning("log in init")

    device = 0 if torch.cuda.is_available() else -1
    model = pipeline('fill-mask', model='bert-base-uncased', device=device)
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    
    print("handler print")
    logging.warning("log handler")

    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(prompt)

    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

# @app.handler runs for every call
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


if __name__ == "__main__":
    app.serve()
