#include <torch/extension.h>
torch::Tensor flash_attention_decode(torch::Tensor query, torch::Tensor keys, torch::Tensor values);
torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, float eps);
torch::Tensor rot_embed(torch::Tensor input, torch::Tensor freq);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("flash_attention_decode", torch::wrap_pybind_function(flash_attention_decode), "flash_attention_decode");
m.def("rms_norm", torch::wrap_pybind_function(rms_norm), "rms_norm");
m.def("rot_embed", torch::wrap_pybind_function(rot_embed), "rot_embed");
}