#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include "imagem.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void blur(unsigned char *input, unsigned char *output, int height, int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int new_value = 0;
    int count = 0;
    for(int l = -1; l <= 1; l++){
        for(int k = -1; k <= 1; k++){
            int index = (((i + l )* width) + (j + k));
            if ((index - width >= 0) && (index + width < height*width) &&  (index + 1 < height*width) && (index - 1 >= 0)) {
                new_value += output[index];
                count++;
            }
        }
    }
    new_value /= count;
    output[i * width + j] = new_value;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Uso: blur-image entrada.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    imagem *img = read_pgm(path);
    imagem *output_img = new_image(img->rows, img->cols);

    thrust::device_vector<unsigned char> input(img->pixels, img->pixels + img->total_size );
    thrust::device_vector<unsigned char> output(output_img->pixels, output_img->pixels + output_img->total_size );

    int nrows = img->rows;
    int ncols = img->cols;
    // dentro do main
    dim3 dimGrid(ceil(nrows/16.0), ceil(ncols/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    blur<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), nrows, ncols);

    thrust::host_vector<unsigned char> O(output);
    for(int i = 0; i != O.size(); i++) {
        output_img->pixels[i] = O[i];
    }
    write_pgm(output_img, "blured.pgm");
    std::cout << "DONE!" << std::endl;

    return 0;
}