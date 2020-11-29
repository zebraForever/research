import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

def add(vec1, vec2, out):
  assert len(vec1) == len(vec2), "len must be equal"
  # allocate memory on the device  
  vec1_gpu = cuda.mem_alloc(vec1.nbytes)
  vec2_gpu = cuda.mem_alloc(vec2.nbytes)
  out_gpu = cuda.mem_alloc(out.nbytes)
  # transfer data from cpu to gpu
  cuda.memcpy_htod(vec1_gpu, vec1)
  cuda.memcpy_htod(vec2_gpu, vec2)
  # define kernel
  mod = SourceModule(r"""
    __global__ void vec_add(int *vec1, int *vec2, int *out) {
      int idx = threadIdx.x;
      out[idx] = vec1[idx] + vec2[idx];
    }
  """)
  vec_add = mod.get_function("vec_add")
  vec_add(vec1_gpu, vec2_gpu, out_gpu, block=(len(vec1), 1, 1))
  # transfer data from gpu to cpu
  cuda.memcpy_dtoh(out, out_gpu)
  # free memoty 
  vec1_gpu.free()
  vec2_gpu.free()
  out_gpu.free()
  return out

def main():
  vec1 = np.array([2, 1, 5, 6, 7]).astype(np.int32)
  vec2 = np.array([6, 2, 1, 8, 1]).astype(np.int32)
  out = np.empty_like(vec1) 
  out_gpu = add(vec1, vec2, out)
  print(out)

if __name__ == "__main__":
  main()
