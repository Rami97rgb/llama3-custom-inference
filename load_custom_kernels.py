from torch.utils.cpp_extension import load
import os

os.environ['TORCH_USE_CUDA_DSA'] = "1"
# suported GPU architectures
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0 8.6 8.7 8.9 9.0"

# simple way to load the CUDA kernel as a pyTorch extension
flash_attention_rami_extension = load(
    name='custom_kernels_extension',
    sources=["./custom_kernels/custom_kernels.cu", "./custom_kernels/main.cpp"],
    with_cuda=True,
    extra_cuda_cflags=[""],
    build_directory='./custom_kernels',
    is_python_module=False,
)