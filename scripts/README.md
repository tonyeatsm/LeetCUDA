# Fork
```bash

# 注意：这里替换成你Fork后的仓库地址
git clone git@github.com:tonyeatsm/LeetCUDA.git
cd LeetCUDA

# 添加原项目为上游（upstream），方便以后同步官方的更新
git remote add upstream git@github.com:tonyeatsm/LeetCUDA.git

```
# 使用Docker挂载你的代码仓库
```bash

sudo docker run --gpus all -itd --name leetcuda \
-v /data/github-workspace/LeetCUDA:/workspace/LeetCUDA \
--user root \
-w /workspace/LeetCUDA \
nvcr.io/nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04

# Enter
sudo docker start leetcuda 
sudo docker exec -it leetcuda /bin/bash

# commit
sudo docker commit leetcuda leetcuda:20260907

# 在容器内初始化并开发
cd /workspace/LeetCUDA
# 更新子模块（这一步在你的宿主机上执行也可以）
apt update && apt install -y git
git config --global --add safe.directory /workspace/LeetCUDA
git submodule update --init --recursive --force && cd kernels/interview





```

# 管理与同步代码
```bash

# 代码修改与提交 在容器内或宿主机上都可以操作
git add .
git commit -m "feat: 完成了elementwise的学习，添加了注释"
git push origin main  # 推送到你自己的GitHub仓库

# 同步官方更新
# 从上游（官方仓库）拉取最新代码
git fetch upstream
# 合并到你的本地分支（以main为例）
git checkout main
git merge upstream/main
# 解决可能出现的冲突（如果有）
# 推送到你自己的远程仓库
git push origin main

# 子模块更新：如果官方更新了子模块（如CUTLASS）
git submodule update --init --recursive --force

```

# Quick Start
```bash
# Enter
sudo docker start leetcuda 
sudo docker exec -it leetcuda /bin/bash

# Find installed cudnn
apt list --installed | grep cudnn

cd /workspace/LeetCUDA/kernels/interview
# Build for target architecture (ccache accelerated when available):
./build.sh --arch sm_89     # Ada Lovelace (L20, RTX 40 series, CUDA Toolkit >= 13.2)
./build.sh --arch sm_90a    # Hopper (H100/H200, CUDA Toolkit >= 13.2)
./build.sh --arch sm_120a   # Blackwell (RTX 5090 / PRO 5000/6000, CUDA Toolkit >= 13.2)
./build.sh --arch all       # All three architectures (sm_89, sm_90a, sm_120a)
./build.sh --clean          # Remove build artifacts (*.o, *.bin, *.ptx)


./notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128 

```

# 安装Python
```bash

# Enter
sudo docker start leetcuda 
sudo docker exec -it leetcuda /bin/bash

export HTTPS_PROXY=http://172.17.0.1:7897
export HTTP_PROXY=http://172.17.0.1:7897

apt update
apt install -y wget curl jq
apt install -y python3.12-dev python3-venv # 选5, 69时区
python3 --version

python3 -m venv /workspace/LeetCUDA/.venv
source /workspace/LeetCUDA/.venv/bin/activate
pip install --upgrade pip wheel
pip install torch==2.14.0 torchvision==0.29.0 torchcodec==0.16.0
pip install ninja==1.13.2

```

# Easy
```bash
sudo docker start leetcuda 
sudo docker exec -it leetcuda /bin/bash
source /workspace/LeetCUDA/.venv/bin/activate

# 打印显卡算力和显卡型号
cd /workspace/LeetCUDA/kernels/elementwise
python3 -c "import torch; name=torch.cuda.get_device_name(); cc=torch.cuda.get_device_capability(); print(f'GPU = {name}'); print(f'Compute Capability = {cc[0]}.{cc[1]} (sm_{cc[0]}{cc[1]})')"

# 逐元素
cd /workspace/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Blackwell # Ada
python3 elementwise.py

# 直方图统计
cd /workspace/LeetCUDA/kernels/histogram
export TORCH_CUDA_ARCH_LIST=Blackwell # Ada
python3 histogram.py

```
