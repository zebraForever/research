import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

def mul(mat1, mat2, out):
  assert mat1.shape[1] == mat2.shape[0], "mat1.shape[1] must be equal mat2.shape[0]"
  # allocate memory on the device
  mat1_gpu = cuda.mem_alloc(mat1.nbytes)
  mat2_gpu = cuda.mem_alloc(mat2.nbytes)
  out_gpu = cuda.mem_alloc(out.nbytes)
  # transert data to device
  cuda.memcpy_htod(mat1_gpu, mat1)
  cuda.memcpy_htod(mat2_gpu, mat2)
  ROWS = mat1.shape[0]
  COLS = mat2.shape[1]
  # define kernel
  ker = r"""
    __global__ void mat_mul(float *mat1, float *mat2, float *out) {
      const int idx = threadIdx.x + blockDim.x * blockIdx.x;
      const int idy = threadIdx.y + blockDim.y * blockIdx.y;
      float res = 0.f;
      for (int i = 0; i < %(COLS)s; i++) {
        res += mat1[idx * %(ROWS)s + i] * mat2[%(COLS)s * i + idy];
      }
      out[idx * %(COLS)s + idy] = res;
    }
  """ % {"ROWS": ROWS,
         "COLS": COLS}
  
  mod = SourceModule(ker)
  mat_mul = mod.get_function("mat_mul")
  mat_mul(mat1_gpu, mat2_gpu, out_gpu, block=(3, 3, 1))
  cuda.memcpy_dtoh(out, out_gpu)
  # free memory
  mat1_gpu.free()
  mat2_gpu.free()
  out_gpu.free()
  return out

def main():
  mat1 = np.array([[3, 5, 1],
                   [6, 1, 4],
                   [8, 5, 2]]).astype(np.float32)
  mat2 = np.array([[6, 3, 1],
                   [7, 4, 4],
                   [2, 2, 4]]).astype(np.float32)
  out = np.empty(shape=(mat1.shape[0], mat2.shape[1])).astype(np.float32)
  
  out = mul(mat1, mat2, out)
  print(out)

if __name__ == "__main__":
  main()
