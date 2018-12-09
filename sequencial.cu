#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;


struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

__global__ void edge_filter(unsigned char *input, unsigned char *output, int rowEnd, int colEnd) {
    int rowStart = 0;
    int colStart = 0;
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    int j=blockIdx.y * blockDim.y + threadIdx.y;
    int di, dj;    
    if (i< rowEnd && j< colEnd) {
        int min = 256;
        int max = 0;
        for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
               if(min>input[di*(colEnd-colStart)+dj]) min = input[di*(colEnd-colStart)+dj];
               if(max<input[di*(colEnd-colStart)+dj]) max = input[di*(colEnd-colStart)+dj]; 
            }
        }
        output[i*(colEnd-colStart)+j] = max-min;
    }
}

result_sssp SSSP(imagem *img, std::vector<int> sources) {
    std::priority_queue<custo_caminho, std::vector<custo_caminho>, compare_custo_caminho > Q;
    double *custos = new double[img->total_size];
    int *predecessor = new int[img->total_size];
    bool *analisado = new bool[img->total_size];

    result_sssp res(custos, predecessor);
    
    for (int i = 0; i < img->total_size; i++) {
        predecessor[i] =-1;
        custos[i] = __DBL_MAX__;
        analisado[i] = false;
    }

    for (int i = 0; i < sources.size(); i++){
        Q.push(custo_caminho(0.0, sources[i]));
        predecessor[sources[i]] = sources[i];
        custos[sources[i]] = 0.0;
    }

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int vertex_i = vertex / img->cols;
        int vertex_j = vertex % img->cols;
        
        if (vertex_i > 0) {
            int acima = vertex - img->cols;
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        if (vertex_i < img->rows - 1) {
            int abaixo = vertex + img->cols;
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        if (vertex_j < img->cols - 1) {
            int direita = vertex + 1;
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        if (vertex_j > 0) {
            int esquerda = vertex - 1;
            double custo_esquerda = custo_atual + get_edge(img, vertex, esquerda);
            if (custo_esquerda < custos[esquerda]) {
                custos[esquerda] = custo_esquerda;
                Q.push(custo_caminho(custo_esquerda, esquerda));
                predecessor[esquerda] = vertex;
            }
        }
    }
    
    delete[] analisado;
    
    return res;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *input_img = read_pgm(path);
    imagem *img = read_pgm(path);

    cudaEvent_t total_start, total_stop, start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    float elapsed_time_edge, elapsed_time_sssp, elapsed_time_seg_img, elapsed_time_total;

    cudaEventRecord(total_start);
    cudaEventRecord(start);

    dim3 dimGrid(ceil(input_img->rows/16.0), ceil(input_img->cols/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    thrust::device_vector<unsigned char> input(input_img->pixels, input_img->pixels + input_img->total_size );
    thrust::device_vector<unsigned char> edge(input_img->pixels, input_img->pixels + input_img->total_size );

    edge_filter<<<dimGrid,dimBlock>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(edge.data()), input_img->rows, input_img->cols);

    thrust::host_vector<unsigned char> O(edge);
    for(int i = 0; i != O.size(); i++) {
        img->pixels[i] = O[i];
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_edge, start, stop);
    // write_pgm(img, "edge_selected.pgm");

    int n_fg, n_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;
    
    std::vector<int> seeds_bg;
    std::vector<int> seeds_fg;

    // CALCULATE DISTANCE TO FG NODE
    for(int i = 0; i < n_bg; i++){
        std::cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }
    for(int i = 0; i < n_fg; i++){
        std::cin >> x >> y;
        int seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
    }
    
    cudaEventRecord(start);
    result_sssp fg = SSSP(img, seeds_fg);
    result_sssp bg = SSSP(img, seeds_bg);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_sssp, start, stop);
    
    imagem *saida = new_image(img->rows, img->cols);
    
    cudaEventRecord(start);
    for (int k = 0; k < saida->total_size; k++) {
        if (fg.first[k] > bg.first[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_seg_img, start, stop);

    write_pgm(saida, path_output);    

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&elapsed_time_total, total_start, total_stop);

    std::cout << elapsed_time_edge << std::endl;
    std::cout << "0" << std::endl;
    std::cout << elapsed_time_sssp << std::endl;
    std::cout << elapsed_time_seg_img << std::endl;
    std::cout << elapsed_time_total << std::endl;

    return 0;
}