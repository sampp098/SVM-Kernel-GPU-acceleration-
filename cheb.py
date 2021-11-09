# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy
import pdb

def Chebyshev_Tn(x,y,n=3,f='r'):
    m=len(x)
    chebyshev_result = 0
    p_cheb = 1
    
    for j in range(0,m):    
        for i in range(0,n+1):
            a= np.cos(i * np.arccos(x[j]))
            b= np.cos(i * np.arccos(y[j]))
            chebyshev_result += (a * b)
        weight = np.sqrt(1 - (x[j] * y[j]) + 0.0002)
        p_cheb *= chebyshev_result / weight 
        #pdb.set_trace()
    return  p_cheb

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pdb

mod = SourceModule("""
__global__ void Chebyshev(float *x, float *y, float *p_cheb){
    int i, j;
    float a,b, chebyshev_result, weight;
    chebyshev_result =0 ;
    p_cheb[0] = 1;
    int idx = blockIdx.x  blockDim.x + threadIdx.x *4;
    for( j=0; j<4; j++){    
        for( i=0; i<4; i++){
            a = cos(i * acos(x[j]));
            b = cos(i * acos(y[j+idx]));
            chebyshev_result += (a * b);
        }
        weight = sqrt(1 - (x[j] * y[j+idx]) + 0.0002);
        p_cheb[0] *= chebyshev_result / weight;
      }
    }
""")
func = mod.get_function("Chebyshev")

def Chebyshev_Tn_gpu(x,y,n=3,f='r'):
    p_cheb = np.random.randn(1,1)
    p_cheb = p_cheb.astype(np.float32)
    p_cheb_gpu = cuda.mem_alloc(p_cheb.nbytes)
    cuda.memcpy_htod(p_cheb_gpu, p_cheb)
    
    t1 = x.astype(np.float32)
    x_gpu = cuda.mem_alloc(t1.nbytes)
    cuda.memcpy_htod(x_gpu, t1)
    
    t2 = y.astype(np.float32)
    y_gpu = cuda.mem_alloc(t2.nbytes)
    cuda.memcpy_htod(y_gpu, t2)


    func(x_gpu,y_gpu,p_cheb_gpu, block=(1,1,1))
    p_cheb = np.empty_like(p_cheb)
    cuda.memcpy_dtoh(p_cheb, p_cheb_gpu)

    return p_cheb