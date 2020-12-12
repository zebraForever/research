import numpy as np
import unittest 
import random
import src
from numpy import testing 
from src.mat_mul import mul

class TestMatMul(unittest.TestCase):
  @staticmethod
  def test_equal_one():
    # test square matrix mul
    for _ in range(10):
      dim_size = random.randint(1, 10)
      mat1 = np.random.randint(100, size=(dim_size, dim_size)).astype(np.float32)
      mat2 = np.random.randint(100, size=(dim_size, dim_size)).astype(np.float32)
      out = np.empty(shape=(mat1.shape[0], mat2.shape[1])).astype(np.float32)
      testing.assert_equal(mul(mat1, mat2, out), mat1 @ mat2)
  
  @staticmethod
  def test_equal_two():
    # test mul mat when the first matrix is square, second is not square
    # and second dim (mat2) more than first dim 
    for _ in range(10):
      dim_size = random.randint(1, 10)
      second_dim_size = random.randint(dim_size, 10)
      mat1 = np.random.randint(100, size=(dim_size, dim_size)).astype(np.float32)
      mat2 = np.random.randint(100, size=(dim_size, second_dim_size)).astype(np.float32)
      out = np.empty(shape=(mat1.shape[0], mat2.shape[1])).astype(np.float32)
      testing.assert_equal(mul(mat1, mat2, out), mat1 @ mat2)  
       
if __name__ == "__main__":
  unittest.main()
