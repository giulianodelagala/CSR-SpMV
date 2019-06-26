#include "io.h"
#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std::chrono;

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define IN_TILE_ROW_SIZE 12
#define IN_TILE_COL_SIZE 12
#define IN_TILE_SLICE_SIZE 12

cudaError_t err = cudaSuccess;

void ImpError(cudaError_t err)
{
	cout << cudaGetErrorString(err); // << " en " << __FILE__ << __LINE__;
	exit(EXIT_FAILURE);
}

//SPMM - Heavy Row Segments
__global__
void parallel_spmv_heavy(int tb_idx, int tb_idy, int tid, int tb_size,
	float** sm_input_value, float** input_value, int* seg_start_num, int* start_seg_position,
	int* seg_index, float* seg_value, int* seg_row_position, float** dest_value)
{
	int index_buf, value_buf, row_idx;
	int row_offset = tb_idx * IN_TILE_ROW_SIZE;
	int slice_offset = tb_idy * IN_TILE_SLICE_SIZE;
	int warp_id = tid / WARP_SIZE;
	int lane_id = tid % WARP_SIZE;
	for (int i = warp_id; i < IN_TILE_ROW_SIZE; i+=tb_size / WARP_SIZE)
	{
		sm_input_value[i][lane_id] =
			input_value[row_offset + i][slice_offset + lane_id];
	}
	__syncthreads();
	
	for (int i = seg_start_num[tb_idx]; i < seg_start_num[tb_idx + 1]; i+= tb_size / WARP_SIZE)
	{
		int val = 0;
		int start = start_seg_position[i];
		int end = start_seg_position[i + 1];
		for (int j = start; j < end - 1; ++j)
		{
			int mod = (j - start) % WARP_SIZE;
			if (mod == 0)
			{
				index_buf = seg_index[j + lane_id];
				value_buf = seg_value[j + lane_id];
			}
				val += sm_input_value[__shfl(index_buf, mod)][lane_id]
				* __shfl(value_buf, mod);
		}
			row_idx = seg_row_position[i];
		// Directamente acumular resultados en memoria global
		atomicAdd(&dest_value[row_idx][slice_offset + lane_id], val);
	}
}

//SPMM - Light Row Segments
__global__
void parallel_spmv_light(int tb_idx, int tb_idy, int tid, int tb_size,
	float** sm_input_value, float** input_value, int* seg_start_num, int* start_seg_position,
	int* seg_index, float* seg_value, float** dest_value)
{
	int* csr_row_pointer, *csr_column_idx, *csr_column_val;
	int index_buf, value_buf;
	int row_offset = (tb_idx * tb_size + tid) / WARP_SIZE;
	int slice_offset = tb_idy * IN_TILE_COL_SIZE;
	int lane_id = tid % WARP_SIZE;
	int start = csr_row_pointer[row_offset];
	int end = csr_row_pointer[row_offset + 1];
	int val = 0;
	for (int i = start; i < end; ++i)
	{
		int mod = (i - start) % WARP_SIZE;
		if (mod == 0)
		{
			index_buf = csr_column_idx[i + lane_id];
			value_buf = csr_column_val[i + lane_id];
		}
		val += input_value[__shfl(index_buf, mod)][lane_id];
		__shfl(value_buf, mod);
	}
	// Directamente acumular resultados en memoria global
	atomicAdd(&dest_value[row_offset][slice_offset + lane_id], val);
}

// SpMV Simple
__global__
void parallel_spmv_1(float* values, int* col_idx, int* row_off, float* vect, float* res,
	int m, int n, int nnz) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m) {
		int begin_index = row_off[row];
		int end_index = row_off[row + 1];

		float row_sum = 0.0;
		for (int i = begin_index; i < end_index; i++) {
			row_sum += (values[i] * vect[col_idx[i]]);
		}

		res[row] = row_sum;
	}

}
////////////////////////////


// TODO SpMV ACSR
__global__
void parallel_spmv_2(float* values, int* col_idx, int* row_off, float* vect, float* res,
	int m, int n, int nnz) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = thread_id / 32;
	int lane_id = thread_id % 32;

	int row = warp_id;

	if (row < m) {
		int begin_index = row_off[row];
		int end_index = row_off[row + 1];

		float thread_sum = 0.0;
		for (int i = begin_index + lane_id; i < end_index; i += 32)
			thread_sum += values[i] * vect[col_idx[i]];

		thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 16);
		thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 8);
		thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 4);
		thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 2);
		thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, 1);

		if (lane_id == 0)
			res[row] = thread_sum;

	}
}
////////////////////////////

// kernel SpMV CSR
__global__
void parallel_spmv_3(float* values, int* col_idx, int* row_off, float* vect, float* res,
	int m, int n, int nnz, int threads_per_row) {

	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int vector_id = thread_id / threads_per_row;
	int lane_id = thread_id % threads_per_row;

	int row = vector_id;

	if (row < m) {
		int begin_index = row_off[row];
		int end_index = row_off[row + 1];

		float thread_sum = 0.0;
		for (int i = begin_index + lane_id; i < end_index; i += threads_per_row)
			thread_sum += values[i] * vect[col_idx[i]];

		int temp = threads_per_row / 2;
		while (temp >= 1) {
			thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, temp);
			temp /= 2;
		}

		if (lane_id == 0)
			res[row] = thread_sum;

	}
}

int nearest_pow_2(float n) {
	int lg = (int)std::log2(n);
	return (int)pow(2, lg);
}

int main(int argc, char** argv) {

	// Creando eventos CUDA //
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	////////////////////////////
	
	int m, n, nnz, nnz_max, nnz_avg, nnz_dev;
	string d(argv[1]);

	// Leyendo Dataset //
	conv(d,nnz, m, n, nnz_max, nnz_avg, nnz_dev); 
	
	//impresion de caracteristicas de la matriz

	cout << "Caracteristicas de " << d;
	cout << "\nnum. filas    = " << m;
	cout << "\nnum. columnas = " << n;
	cout << "\nnun. non-zero = " << nnz;
	cout << "\nmax. non-zero por fila = " << nnz_max;
	cout << "\npromedio de non-zero por fila = " << nnz_avg;
	cout << "\ndesviacion de non-zero = " << nnz_dev;
	cout << "\n\n";

	float* vect = GenDenseVector(n); //generar vector denso
	////////////////////////////


	// Serial SpMV //
	float* host_res = new float[m];
	//Toma de tiempo CPU
	auto ini = high_resolution_clock::now();
	MultiSimpleSPMV(host_res, vect, valores, col_idx, row_offset, nnz, m, n);
	auto fin = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(fin - ini);
	////////////////////////////


	// Asignacion de Device Memory 
	float* d_valor, *d_res, *d_vect;
	int* d_row_offset, *d_col_index;

	err = cudaMalloc((void**)& d_valor, sizeof(float) * nnz);
	err = cudaMalloc((void**)& d_col_index, sizeof(int) * nnz);
	err = cudaMalloc((void**)& d_row_offset, sizeof(int) * (m + 1));
	err = cudaMalloc((void**)& d_res, sizeof(float) * m);
	err = cudaMalloc((void**)& d_vect, sizeof(float) * n);
	
	if (err != cudaSuccess)
		ImpError(err);

	// Copia de Host a device //
	err = cudaMemcpy(d_valor, valores, sizeof(float) * nnz, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_col_index, col_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_row_offset, row_offset, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vect, vect, sizeof(float) * n, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess)
		ImpError(err);
	////////////////////////////


	// SpMV Paralelos en CUDA
	//Asignacion de de grilla y bloque segun algoritmo

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid_1((m - 1) / BLOCK_SIZE + 1, 1, 1);
	dim3 dimGrid_2((m - 1) / 32 + 1, 1, 1);
	int threads_per_row = min(32, nearest_pow_2(nnz_avg));
	dim3 dimGrid_3((m - 1) / (1024 / threads_per_row) + 1, 1, 1);

	// LLamada a kernel SpMV simple
	cudaEventRecord(start);
	parallel_spmv_1 <<<dimGrid_1, dimBlock >>> 
		 (d_valor, d_col_index, d_row_offset, d_vect, d_res, m, n, nnz);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpu_time_1 = 0;
	cudaEventElapsedTime(&gpu_time_1, start, stop);

	// TODO Llamada a kernel ACSR
	
	cudaEventRecord(start);
	parallel_spmv_2 << <dimGrid_2, dimBlock >> > 
		(d_valor, d_col_index, d_row_offset, d_vect, d_res, m, n, nnz);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpu_time_2 = 0;
	cudaEventElapsedTime(&gpu_time_2, start, stop);
	
	// LLama a kernel CSR
	cudaEventRecord(start);
	parallel_spmv_3 << <dimGrid_3, dimBlock >> > 
		(d_valor, d_col_index, d_row_offset, d_vect, d_res, m, n, nnz, threads_per_row);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float gpu_time_3 = 0;
	cudaEventElapsedTime(&gpu_time_3, start, stop);

	// Copia de Resultados desde device a host
	float* result_from_device = new float[m];
	err = cudaMemcpy(result_from_device, d_res, sizeof(float) * n, cudaMemcpyDeviceToHost);

	// Comprobar resultados de CUDA con CPU
	Comprobar(result_from_device, host_res, m);

	// Liberar memoria
	cudaFree(d_valor);
	cudaFree(d_col_index);
	cudaFree(d_row_offset);
	cudaFree(d_res);
	cudaFree(d_vect);

	// Impresion de tiempos 
	cout << "\nTiempo de Ejecucion Serial en CPU = " << duration.count() << " ms";
	
	cout << "\nTiempo de Ejecucion en GPU = " << gpu_time_1 << " ms";

	cout << "\nTiempo de Ejecucion en GPU = " << gpu_time_2 << " ms";
	
	cout << "\nTiempo de Ejecucion en GPU formato CSR   = " << gpu_time_3 << " ms";

	if (err != cudaSuccess)
		ImpError(err);

	return 0;
}