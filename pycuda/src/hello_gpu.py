import pycuda.autoinit
from pycuda.compiler import SourceModule

def main():
  mod = SourceModule(r"""
    #include <stdio.h>

    __global__ void hello_gpu() {
      int i = threadIdx.x;
      printf("Hello from %d thread!\n", i);
    }
    """)

  func = mod.get_function("hello_gpu")
  func(block=(4, 1, 1))

if __name__ == "__main__":
  main()
