__kernel void multi(__global const float* values,
	__global const int* col_idx,
	__global int* row_off,
	__global float* vect,
	__global float* res,
	__global int m,
	__global int n,
	__global int nnz)

{

	// Indice a ser procesado
	int row = get_global_id(0);

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