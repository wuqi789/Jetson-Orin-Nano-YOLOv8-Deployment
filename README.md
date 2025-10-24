# Jetson Orin Nano 部署并运行 YOLOv8（YOLO11 同理）

## 步骤概述

在 NVIDIA Jetson Orin Nano 上部署并运行 YOLOv8（或 YOLO11）的详细步骤。

## 1. 下载并烧录 JetPack 系统

- 在 NVIDIA 官方下载 [SDK Manager](https://developer.nvidia.com/sdk-manager)（SDK Manager | NVIDIA Developer）进行烧录纯净 JetPack 系统（或确保系统和环境已经配置好）。

- 注意 Ubuntu 版本和 JetPack 版本的兼容性（推荐使用ubuntu20.04进行烧录，烧录前内存至少8GB,硬盘至少80GB）。

  ![这是图片](/images/1.jpg "1")

- 推荐使用 JetPack 6.1，它可以使性能比肩 Super 版本，达到 67 TOPS。（截至2025年1月5日nvidia官网[NVIDIA 开发者论坛 PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)暂无pytorch for jetson 的jp6.1 v1版本）

- 直接烧录系统，包括 CUDA、cuDNN、TensorRT 和 OpenCV。

## 2. 安装 JetPack 和 Jtop

- 开机后安装 JetPack：
  ```bash
  sudo apt update
  sudo apt install nvidia-jetpack
  ```
  
- 安装 Jtop（用于监控 Jetson 设备的性能和资源使用情况）：
  ```bash
  sudo apt-get install python3-pip
  sudo pip3 install -U pip
  sudo pip install -U jetson-stats
  ```
  
- 重启设备：
  
  ```bash
  Reboot
  ```
 
  ![这是图片](/images/2.jpg "1")

## 3. 下载并安装 Miniforge

- 下载 Miniforge：[Miniforge GitHub 页面](https://github.com/conda-forge/miniforge)

  ```bash
  git clone https://github.com/conda-forge/miniforge.git
  ```

- 选择 Linux aarch64（arm64）版本，运行 shell（.sh）文件，再两次选择 yes 完成安装（如果因为网络问题无法下载，建议在自己电脑下载好到U盘里再传到jetson orin nano）。

## 4. 创建并激活虚拟环境

- 创建名为 `yolov8` 的虚拟环境，Python 版本为 3.8：
  ```bash
  conda create -n yolov8 python=3.8
  ```
- 激活环境：
  ```bash
  conda activate yolov8
  ```

## 5. 安装 Ultralytics、ONNX、LAPX 和 NumPy

- 使用清华源安装 Ultralytics、ONNX、LAPX 和 NumPy（版本 1.23.1）：
  ```bash
  pip install ultralytics onnx lapx numpy==1.23.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

## 6. 安装 PyTorch 和 torchvision

- 根据 JetPack 版本确定所需的 PyTorch 版本，再通过 PyTorch 版本找到对应的 torchvision 版本。
  - JetPack 对应 PyTorch：[NVIDIA 开发者论坛 PyTorch for Jetson 页面](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
  - PyTorch 对应 torchvision：[PyTorch Vision GitHub 页面](https://github.com/pytorch/vision)
- 安装 torchvision 依赖：
  ```bash
  sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
  pip3 install setuptools
  ```
- 根据 JetPack 和 PyTorch 版本安装相应的 PyTorch 和 torchvision。

## 7. 检查环境

- 进入虚拟环境，检查已安装的包：
  ```bash
  pip list
  ```
  
  ![这是图片](/images/3.jpg "1")
  
- 尝试导入 torch和 torchvision 以验证环境配置：
  
  ```python
  import torch
  import torchvision
  ```

![这是图片](/images/5.jpg "1")

## 8. 创建 YOLO 文件夹并准备权重文件

- 创建 `yolo` 文件夹：
  ```bash
  cd ~
  mkdir yolo
  ```
- 将 YOLOv8（或 YOLOv11）训练的 Ultralytics 和 `.pt` 文件粘贴到 `yolo` 文件夹中。

## 9. Tensorrtx 检测

- 克隆 Tensorrtx 仓库：
  ```bash
  git clone https://github.com/wang-xinyu/tensorrtx.git
  ```
  
- 切换到 YOLOv8 目录，并准备构建文件：（若为yolo11就进入yolo11文件夹）
  ```bash
  cd tensorrtx/yolov8
  mkdir build
  cp yolov8_wts.py ../yolo
  cd ../yolo
  ```
  
- 使用 `gen_wts.py` 脚本将 `.pt` 文件转换为 `.wts` 文件：

  ```bash
  python gen_wts.py -w yolov8n.pt -o yolov8n.wts -t detect  #yolov8运行这个
  python gen_wts.py -w yolo11n.pt -o yolo11n.wts -t detect  #yolo11运行这个
  ```

  若出错可能是环境变量的问题，参考环境变量如下
  
  ![这是图片](/images/6.jpg "1")
  
- 将生成的 `.wts` 文件复制到 `build` 目录中，并构建 TensorRTX 项目：
  
  ```bash
  cp yolov8n.wts ../yolov8/build/#yolov8用这两行
  cd ../yolov8/build
  cp yolo11n.wts ../yolo11/build/#yolo11用这两行
  cd ../yolo11/build  
  #无论8还是11都需要编译
  cmake ..
  make  #如果更改模型文件需要重新编译
  ```
  
  ![这是图片](/images/4.jpg "1")
  
- 运行 YOLOv8 检测程序，并指定 `.wts` 文件和参数量大小：
  
  ```bash
  sudo ./yolov8_det -s yolov8n.wts yolov8n.engine n #yolov8运行这个
  sudo ./yolo11_det -s yolo11n.wts yolo11n.engine n #yolo11运行这个
  ```
- 得到yolov8n.engine文件后修改并运行yolov8_det_trt.py
- 或
- 得到yolo11n.engine文件后修改并运行yolo11_det_trt.py
（注意：报错是正常的，此时还不能实现实时检测。需要更改检测方法）

## 10. 修改 YOLOv8 检测文件

- 根据需要修改 `yolo_det_trt.py`（用于 Python 检测）或 `yolo_det.cpp`（用于 C++ 检测）文件，以适应您的具体应用场景。
直接替换即可，然后在文件中的main里确认yolov8n.engine的路径，以及修改类别categories，最后直接运行yolo_det_trt.py即可
  （相关文件已上传）

  #### ***Written By WuQi***
  
