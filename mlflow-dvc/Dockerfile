# 1. Use the official Ultralytics image pre-configured with CUDA, PyTorch, and OpenCV
FROM ultralytics/ultralytics:latest

# 2. Set an explicit working directory inside the container virtual file system
WORKDIR /workspace

# 3. Copy only the requirements first to take advantage of Docker layer caching
COPY requirements.txt .

# 4. Install MLflow and cloud utilities 
RUN pip install -r requirements.txt

# 5. Copy your local script files and default configurations into the workspace
COPY . /workspace/

