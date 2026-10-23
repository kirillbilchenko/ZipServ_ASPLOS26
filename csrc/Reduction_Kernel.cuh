// Modified 2026 by Kyrylo Bilchenko: correctness and MxWave backports.
/***************************************************************************
 * Copyright 2025 The ZipServ Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#define ELEMENT_PER_THREADBLOCK 256
// Split-K reduction operation
__global__ void SplitK_Reduction_BF16(__nv_bfloat16* C, __nv_bfloat16* Reduction_Workspace, 
                                   int M_Global, int N_Global, int Split_K)
{
    __nv_bfloat16* C_BasePTR_ThisBlock = C + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    __nv_bfloat16* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    
    float Results[HALF_PER_128B];
    
    #pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    
    for (int i = 0; i < Split_K; i++) {
        #pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            if (ELEMENT_PER_THREADBLOCK * blockIdx.x + threadIdx.x * HALF_PER_128B + j < M_Global * N_Global)
                Results[j] += __bfloat162float(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }
    
    #pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        if (ELEMENT_PER_THREADBLOCK * blockIdx.x + threadIdx.x * HALF_PER_128B + j < M_Global * N_Global)
            C_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j] = __float2bfloat16(Results[j]);
}

