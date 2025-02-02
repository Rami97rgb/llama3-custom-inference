#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// set precision dtype, only blfoat16 is supported for now (used by llama 3 models)
typedef __nv_bfloat16 Precision;

// vector size for loading kv cache from global mem
const int VEC_SIZE = 16 / sizeof(Precision);

template <typename T, int VEC_SIZE>
struct Vec {};
template <>
struct Vec<__nv_bfloat16, 2> {
  using Type = __nv_bfloat162;
};
struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};
template <>
struct Vec<__nv_bfloat16, 4> {
  using Type = bf16_4_t;
};
struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};
template <>
struct Vec<__nv_bfloat16, 8> {
  using Type = bf16_8_t;
};
using Precision_Vec = typename Vec<Precision, VEC_SIZE>::Type;

// convert kernel dtype to torch dtype
template <typename T>
struct Precision2Torch {};
template <>
struct Precision2Torch<__nv_bfloat16> {
  using Type = at::BFloat16;
};
using P2Torch = typename Precision2Torch<Precision>::Type;

// convert dtype from and to float32
template<typename T> inline __device__  T float2p(const float);
template <> __nv_bfloat16 float2p(const float a){
    return  __float2bfloat16(a);
}

inline __device__ float p2float(const __nv_bfloat16 a){
    return   __bfloat162float(a);
}
