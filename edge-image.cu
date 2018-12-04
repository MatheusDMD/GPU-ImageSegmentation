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

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

using namespace std;

_global_ void edge(unsigned char *in, unsigned char *out, int rowStart, int rowEnd, int colStart, int colEnd){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    int di, dj;    
    if (i< rowEnd && j< colEnd) {
        int min = 256;
        int max = 0;
        for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
               if(min>in[di*(colEnd-colStart)+dj]) min = in[di*(colEnd-colStart)+dj];
               if(max<in[di*(colEnd-colStart)+dj]) max = in[di*(colEnd-colStart)+dj]; 
            }
        }
        out[i*(colEnd-colStart)+j] = max-min;
    }
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
    // blur<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), nrows, ncols);
    edge<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), 0, img->rows, 0, img->cols);
    thrust::host_vector<unsigned char> O(output);
    for(int i = 0; i != O.size(); i++) {
        output_img->pixels[i] = O[i];
    }
    write_pgm(output_img, "blured.pgm");
    std::cout << "DONE!" << std::endl;

    return 0;
}