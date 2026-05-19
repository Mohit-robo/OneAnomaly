## 1. Start the Triton Inference Server

Open a new terminal window, navigate to your project directory, and start the Triton Docker container. We must map your host's triton_models folder to the container's /models directory and expose the gRPC/HTTP ports.

    cd /home/silicon/projects/anomaly_app
    
    docker run --gpus all --rm \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        -v $(pwd)/triton_models:/models \
        nvcr.io/nvidia/tritonserver:24.08-py3 \
        tritonserver --model-repository=/models

(Wait until you see Started GRPCInferenceService at 0.0.0.0:8001 and Started HTTPService at 0.0.0.0:8000)

    docker run --gpus all --rm -v /home/silicon/projects/anomaly_app/models/onnx_models:/models -v /home/silicon/projects/anomaly_app/triton_models/dinov3_encoder/1:/out nvcr.io/nvidia/tritonserver:24.08-py3 
    
    /usr/src/tensorrt/bin/trtexec --onnx=/models/dinov3_vits16_dynamic.onnx --saveEngine=/out/model.plan --minShapes=input:1x3x224x224 --optShapes=input:4x3x224x224 --maxShapes=input:8x3x224x224

## 2. Start the Local GPU Edge Gateway

Open a second terminal. This FastAPI server will broker raw images from the cloud server, run local PyTorch/OpenCV operations, and talk to Triton rapidly via loopback.


    # Activate your local virtual environment
    source /home/silicon/projects/anomaly_app/anomaly_app/bin/activate
    
    # Navigate to the gateway directory and run the Uvicorn server
    cd /home/silicon/projects/anomaly_app/gateway
    
    python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
    
(This listens on port 8080 locally)

## 3. Start the Cloud API Proxy (Frontend Web Server)

Open a third terminal. This mimics what will run on Google Cloud Run. It launches the UI and safely proxies requests down to the Gateway.

    # Activate your local virtual environment
    source /home/silicon/projects/anomaly_app/anomaly_app/bin/activate
    
    # Navigate to the web API directory and run the Flask server
    cd /home/silicon/projects/anomaly_app/python
    
    python api_server.py