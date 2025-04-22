# 基础镜像，使用与当前环境匹配的CUDA版本
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04


# 设置工作目录
WORKDIR /workspace

# 避免安装过程中的交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置时区（避免交互式提示）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN python3 -m pip install --upgrade pip

# 安装Python包依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建输入输出目录
RUN mkdir -p /workspace/inputs
RUN mkdir -p /workspace/outputs

# 复制模型文件和预测脚本
COPY predict.sh /workspace/
RUN mkdir -p /workspace/model

COPY model/fastsam3d.pth /workspace/model/fastsam3d.pth
COPY model/sam_med3d.pth /workspace/model/sam_med3d.pth

COPY code/ /workspace/
# 设置执行权限
RUN chmod +x /workspace/predict.sh

# 设置工作目录的权限
RUN chmod -R 777 /workspace

