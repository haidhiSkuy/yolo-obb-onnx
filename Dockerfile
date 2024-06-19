FROM python

WORKDIR /app 
COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python onnxruntime fastapi pillow 
RUN pip install numpy==1.26.4


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]