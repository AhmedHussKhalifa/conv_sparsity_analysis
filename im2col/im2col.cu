#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions

#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
using namespace std;

#define CUDA_CALL( call )               \
{                                       \
cudaError_t result = call;              \
if ( cudaSuccess != result )            \
    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;  \
}




__global__ void func_old(float* input,float* output,
							int i_w, int i_h, int k_w,int k_h,int stride,
							int o_w,int o_h){
	const unsigned int thread_idx = blockIdx.x;
	int temp =0;
	int block_size = k_w*k_h; //待改
	for(int i=0;i<k_h;i++){
		for(int j=0;j<k_w;j++){
			output[thread_idx*block_size+temp] = input[thread_idx%o_w*stride+j+i*i_w+thread_idx/o_h*i_w];
			//printf("%d\n",thread_idx*block_size+temp);
			temp++;
		}
	}
}


int main(){
	int fileNum = 2 * 94;
	string a[fileNum];
	int h[fileNum],w[fileNum];
	ifstream file_list("../gen/file_list");
	for(int i=0;i<fileNum;i++)
		file_list>>a[i];
	file_list.close();
	
	ifstream conv_shape("../gen/conv_shape");
	for(int i=0;i<fileNum;i++){
		conv_shape>>h[i];
		w[i] = h[i];
	}
	conv_shape.close();
	
	// per-experiment
	int len;
	const int k_w = 3;
	const int k_h = 3;
	const int stride = 3;
	
	const int kernelSize = k_w*k_h;
	const float kernel[kernelSize] = { 1,0,1,0,1,1,0,1,1};
	
	for(int i=0;i<fileNum;i++){
		int i_w = w[i];
		int i_h = h[i];
		//cout<<i_w<<i_h<<endl;
		int arraySize = i_w*i_h;
		int o_w = (i_w-k_w)/stride +1;
		int o_h = (i_h-k_h)/stride +1;
		int outSize = o_w*o_h*k_w*k_h;
			
		
		float *feature = new float[arraySize];
		
		len = 0;
		ifstream conv_feature(("../conv/"+a[i]).c_str());
		while(!conv_feature.eof())
			conv_feature>>feature[len++];
		conv_feature.close();
	
		float *im2col = new  float[outSize];
	

		float * gpu_input;
		float * gpu_output;

	

		CUDA_CALL(cudaMalloc((void**)&gpu_input,arraySize*sizeof(float)));

		CUDA_CALL(cudaMalloc((void**)&gpu_output,outSize*sizeof(float)));

		CUDA_CALL(cudaMemcpy(gpu_input,feature,arraySize*sizeof(float),cudaMemcpyHostToDevice));

		cudaEvent_t start,stop;
		float elapsedTime1 = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		func_old<<<o_w*o_h,1>>>(gpu_input,gpu_output,i_w,i_h,k_w,k_h,stride,o_w,o_h);
	
		CUDA_CALL(cudaMemcpy(im2col,gpu_output,outSize*sizeof(float),cudaMemcpyDeviceToHost));
	
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime1, start, stop);

		//cout << elapsedTime1<<"ms" << endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(gpu_input);
		cudaFree(gpu_output);

		cublasHandle_t handle;
		cublasCreate(&handle);
	
	
		float * gpu_im2col;
		float * gpu_kernel;
		float * gpu_result;
		float * cpu_result;
		cpu_result = (float*)malloc(sizeof(float)*o_w*o_h);
	
	

		CUDA_CALL(cudaMalloc((void**)&gpu_im2col,outSize*sizeof(float)));
		CUDA_CALL(cudaMalloc((void**)&gpu_kernel,k_w*k_h*sizeof(float)));
		CUDA_CALL(cudaMalloc((void**)&gpu_result,o_w*o_h*sizeof(float)));
		
		CUDA_CALL(cudaMemcpy(gpu_im2col,im2col,outSize*sizeof(float),cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(gpu_kernel,kernel,k_w*k_h*sizeof(float),cudaMemcpyHostToDevice));
	
		float a=1;
		float b=0;
//cudaEvent_t start,stop;
		float elapsedTime2 = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, o_w*o_h, k_w*k_h, &a,gpu_kernel , 1, gpu_im2col, k_w*k_h, &b, gpu_result, 1);
	cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime2, start, stop);

		//cout << elapsedTime2<< endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaMemcpy(cpu_result,gpu_result,o_w*o_h*sizeof(float),cudaMemcpyDeviceToHost);

	


		cout <<elapsedTime1 +elapsedTime2 << endl;
	
		cudaFree(gpu_im2col);
		cudaFree(gpu_kernel);
		cudaFree(gpu_result);


		free(cpu_result);
	}
	return 0;
}

/*
0 1 2 3 4 
5 6 7 8 9 
10 11 12 13 14 
15 16 17 18 19
20 21 22 23 24

0,1,1,0,1,0,0,1,0

0 1 2 5 6 7 10 11 12
1 2 3 6 7 8 11 12 13
2 3 4 7 8 9 12 13 14
5 6 7 10 11 12 15 16 17
6 7 8 11 12 13 16 17 18
7 8 9 12 13 14 17 18 19
10 11 12 15 16 17 20 21 22
11 12 13 16 17 18 21 22 23
12 13 14 17 18 19 22 23 24
*/


/*
float * gpu_feature1;
	float * gpu_kernel1;
	float * gpu_result1;
	float * cpu_result1;
	cpu_result1 = (float*)malloc(sizeof(float)*o_w*o_h);

	CUDA_CALL(cudaMalloc((void**)&gpu_feature1,arraySize*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&gpu_kernel1,k_w*k_h*sizeof(float)));
	CUDA_CALL(cudaMalloc((void**)&gpu_result1,o_w*o_h*sizeof(float)));
	
	CUDA_CALL(cudaMemcpy(gpu_feature1,feature,arraySize*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(gpu_kernel1,kernel,k_w*k_h*sizeof(float),cudaMemcpyHostToDevice));

	func_new<<<2,2>>>(gpu_feature1,gpu_kernel1,gpu_result1,i_w,i_h,k_w,k_h,stride,o_w,o_h);

	CUDA_CALL(cudaMemcpy(cpu_result1,gpu_result1,o_w*o_h*sizeof(float),cudaMemcpyDeviceToHost));

0.352096ms
0.1832ms
0.535296ms
38 44 50 68 74 80 98 104 110 请按任意键继续. . .


*/
