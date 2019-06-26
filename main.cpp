#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "io.h"
#include "util.h"

using namespace std::chrono;

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char** argv)
{
	int m, n, nnz, nnz_max, nnz_avg, nnz_dev;
	string d(argv[1]);

	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;

	fp = fopen("multi.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue_properties properties[] {CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, *properties, &ret);

	// Leyendo Dataset //
	LecturaMTX(d, nnz, m, n, nnz_max, nnz_avg, nnz_dev);

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

	// Serial SpMV //
	float* host_res = new float[m];
	//Toma de tiempo CPU
	auto ini = high_resolution_clock::now();
	MultiSimpleSPMV(host_res, vect, valores, col_idx, row_offset, nnz, m, n);
	auto fin = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(fin - ini);
	////////////////////////////

	// Asignacion de Device Memory 
	//float* d_valor, * d_res, * d_vect;
	//int* d_row_offset, * d_col_index;

	// Crear buffers de memoria para cada vector
	cl_mem d_valor = clCreateBuffer(context, CL_MEM_READ_ONLY,
		nnz * sizeof(float), NULL, &ret);

	cl_mem d_res = clCreateBuffer(context, CL_MEM_READ_ONLY,
		m * sizeof(float), NULL, &ret);

	cl_mem d_vect = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		n * sizeof(float), NULL, &ret);

	cl_mem d_row_offset = clCreateBuffer(context, CL_MEM_READ_ONLY,
		(m + 1) * sizeof(int), NULL, &ret);

	cl_mem d_col_index = clCreateBuffer(context, CL_MEM_READ_ONLY,
		nnz * sizeof(int), NULL, &ret);

	// Copiar vectores a sus buffers
	ret = clEnqueueWriteBuffer(command_queue, d_valor, CL_TRUE, 0,
		nnz * sizeof(float), valores, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, d_col_index, CL_TRUE, 0,
		nnz * sizeof(int), col_idx, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, d_row_offset, CL_TRUE, 0,
		(m + 1) * sizeof(int), row_offset, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, d_vect, CL_TRUE, 0,
		n * sizeof(float), vect, 0, NULL, NULL);

	// Crear un programa desde un kernel
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char**)& source_str, (const size_t*)& source_size, &ret);

	// Construccion del programa
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// Crear OpenCL Kernel
	cl_kernel kernel = clCreateKernel(program, "multi", &ret);

	// Set argumentos de Kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& d_valor);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& d_col_index);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& d_row_offset);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& d_vect);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& d_res);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& m);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& n);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& nnz);

	// Ejecutar el kernel
	size_t global_item_size = m; // Tamanho de la lista a ejecutar
	size_t local_item_size = 64; // Dividir el trabajo
	
	//Crear evento para toma de tiempo
	
	//cl_event event;
	
	auto inicio = high_resolution_clock::now();
	
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_item_size, &local_item_size, 0, NULL, nullptr);
	
	auto final = high_resolution_clock::now();
	auto duration_cl = duration_cast<microseconds>(final - inicio);
	
	//clWaitForEvents(1, &event);

	// Copiar resultado a host
	float* result_from_device = (float*)malloc(sizeof(float) * m);
	ret = clEnqueueReadBuffer(command_queue, d_res, CL_TRUE, 0,
		m * sizeof(float), result_from_device, 0, NULL, NULL);

	
	
	// Comprobar resultados de CUDA con CPU
	Comprobar(result_from_device, host_res, m);

	//Limpieza

	ret = clReleaseMemObject(d_valor);
	ret = clReleaseMemObject(d_res);
	ret = clReleaseMemObject(d_vect);
	ret = clReleaseMemObject(d_row_offset);
	ret = clReleaseMemObject(d_col_index);


	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);

	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	
	free(result_from_device);

	// Impresion de tiempos 
	cout << "\nTiempo de Ejecucion Serial en CPU = " << duration.count() << " ms";

	cout << "\nTiempo de Ejecucion en GPU = " << duration_cl.count() << " ms";


	//ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	//ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

	//double nanoSeconds = time_end - time_start;
	//printf("OpenCl Execution time is: %0.3f milliseconds \n", nanoSeconds / 1000000.0);
	//std::chrono::nanoseconds ss(time_end - time_start);

	//cout << "Tiempo" << ss.count();

	return 0;
}