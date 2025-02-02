#include <torch/extension.h>
torch::Tensor flash_attention_decode(torch::Tensor query, torch::Tensor keys, torch::Tensor values);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("flash_attention_decode", torch::wrap_pybind_function(flash_attention_decode), "flash_attention_decode");
}