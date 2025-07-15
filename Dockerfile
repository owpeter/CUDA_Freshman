# 1. 选择一个包含 CUDA 开发工具的基础镜像
# 格式: nvidia/cuda:<cuda_version>-devel-<os_version>
# 'devel' 镜像是必须的，因为它包含了 nvcc 编译器和库文件。
# 'runtime' 镜像只包含运行时，不能用于编译。
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# 2. 设置工作目录
WORKDIR /app

# 3. 将项目文件从宿主机复制到容器中
# 第一个 '.' 代表宿主机当前目录下的所有文件 (我们的 hello_cuda.cu)
# 第二个 '.' 代表容器的当前工作目录 (/app)
COPY . .