#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include <nvgraph.h>
#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;

typedef std::pair<int, int> seed;

struct graphParams {
    float * weights_h;
    int * destination_offsets_h;
    int * source_indices_h;
    int source_seed;
    size_t n;
    size_t nnz;
};


void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

int NvidiaSSSH(float *weights_h, int *destination_offsets_h, int *source_indices_h, const size_t n, const size_t nnz, int source_seed, float *sssp_1_h) {
    // const size_t  n = 6, nnz = 10;
    const size_t vertex_numsets = 1, edge_numsets = 1;
    // int i; //*destination_offsets_h, *source_indices_h;
    // float *sssp_1_h;// *sssp_2_h; //*weights_h, 
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    
    // sssp_1_h = (float*)malloc(n*sizeof(float));
    //sssp_2_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    // destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
    // source_indices_h = (int*) malloc(nnz*sizeof(int));
    // weights_h = (float*)malloc(nnz*sizeof(float));
    vertex_dim[0]= (void*)sssp_1_h; //vertex_dim[1]= (void*)sssp_2_h;
    vertex_dimT[0] = CUDA_R_32F; //vertex_dimT[1]= CUDA_R_32F;

    // weights_h [0] = 0.333333;
    // weights_h [1] = 0.500000;
    // weights_h [2] = 0.333333;
    // weights_h [3] = 0.500000;
    // weights_h [4] = 0.500000;
    // weights_h [5] = 1.000000;
    // weights_h [6] = 0.333333;
    // weights_h [7] = 0.500000;
    // weights_h [8] = 0.500000;
    // weights_h [9] = 0.500000;

    // destination_offsets_h [0] = 0;
    // destination_offsets_h [1] = 1;
    // destination_offsets_h [2] = 3;
    // destination_offsets_h [3] = 4;
    // destination_offsets_h [4] = 6;
    // destination_offsets_h [5] = 8;
    // destination_offsets_h [6] = 10;

    // source_indices_h [0] = 2;
    // source_indices_h [1] = 0;
    // source_indices_h [2] = 2;
    // source_indices_h [3] = 0;
    // source_indices_h [4] = 4;
    // source_indices_h [5] = 5;
    // source_indices_h [6] = 2;
    // source_indices_h [7] = 3;
    // source_indices_h [8] = 3;
    // source_indices_h [9] = 4;

    std::cout << "SSSP1" << std::endl;
    
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    std::cout << "SSSP2" << std::endl;
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    std::cout << "SSSP3" << std::endl;
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    std::cout << "SSSP4" << std::endl;
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    std::cout << "SSSP5" << std::endl;
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    std::cout << "SSSP6" << std::endl;
    
    // Solve
    int source_vert = source_seed; //source_seed
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    std::cout << "SSSP7" << std::endl;
    
    
    // Solve with another source
    // source_vert = 5;
    // check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 1));
    std::cout << "SSSP8" << std::endl;
    
    // Get and print result
    
    
    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    std::cout << "SSSP9" << std::endl;
    // expect sssp_1_h = (0.000000 0.500000 0.500000 1.333333 0.833333 1.333333)^T
    
    std::cout << "SSSP10" << std::endl;
    
    // check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_2_h, 1));
    // // expect sssp_2_h = (FLT_MAX FLT_MAX FLT_MAX 1.000000 1.500000 0.000000 )^T
    // printf("sssp_2_h\n");
    // for (i = 0; i<n; i++)  printf("%f\n",sssp_2_h[i]); printf("\n");
    // printf("\nDone!\n");
    
    std::cout << "SSSP11" << std::endl;
    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    // free(sssp_1_h);
    // free(sssp_2_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    
    std::cout << "SSSP12" << std::endl;
    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));
    
    return 0;
}

struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

graphParams GetGraphParams(imagem *img, std::vector<int> seeds, int seeds_count){
    std::vector<int> dest_offsets;
    std::vector<int> src_indices;
    std::vector<float> weights;
    
    // params.weights_h = new float[params.n]();
    // params.destination_offsets_h = new int[params.nnz + 1](); // +1 to add the size as the last item to the end of the array
    // params.source_indices_h = new int[params.n]();a
    int count = 0;
    dest_offsets.push_back(0); // add zero to position 0

    std::cout << "start:" << std::endl;
    std::cout << std::endl;
    
    for(int vertex = 0; vertex < img->total_size; vertex++ ){

        int local_count = 0;

        int vertex_i = vertex / img->cols;
        int vertex_j = vertex % img->cols;

        if (vertex_i > 0) {
            int acima = vertex - img->cols;
            double custo_acima = get_edge(img, vertex, acima);
            src_indices.push_back(acima);
            weights.push_back(custo_acima);
            local_count++;
        }
        if (vertex_i < img->rows - 1) {
            int abaixo = vertex + img->cols;
            double custo_abaixo = get_edge(img, vertex, abaixo);
            src_indices.push_back(abaixo);
            weights.push_back(custo_abaixo);
            local_count++;
        }

        if (vertex_j < img->cols - 1) {
            int direita = vertex + 1;
            double custo_direita = get_edge(img, vertex, direita);
            src_indices.push_back(direita);
            weights.push_back(custo_direita);
            local_count++;
        }

        if (vertex_j > 0) {
            int esquerda = vertex - 1;
            double custo_esquerda = get_edge(img, vertex, esquerda);
            src_indices.push_back(esquerda);
            weights.push_back(custo_esquerda);
            local_count++;
        }

        dest_offsets.push_back(dest_offsets.back() + local_count); // add local_count to last position vector
    }

    // for(int i = 0; i < seeds_count; i++ ){
    //     weights.push_back(0);
    //     src_indices.push_back(seeds[i]);
    //     dest_offsets.push_back(dest_offsets.back() + 1);
    //     std::cout << seeds[i] << std::endl;
    // }
    // dest_offsets.push_back(dest_offsets.back());

    graphParams params = {};
    params.nnz = (img->total_size * 4) - (((img->cols + img->rows) * 2));// + seeds_count); //total connections (+ seed_count to add the connections from all nodes from that type)
    // params.source_indices_h = (int*) malloc(params.nnz*sizeof(int));
    // params.weights_h = (float*)malloc(params.nnz*sizeof(float));
    params.source_indices_h = (int*) malloc(params.nnz*sizeof(int));
    params.weights_h = (float*)malloc(params.nnz*sizeof(float));

    params.n = dest_offsets.size() - 1;
    params.destination_offsets_h = (int*) malloc((params.n+1)*sizeof(int));
    
    std::cout << "src_indices: " ;
    for (int index = 0; index < src_indices.size(); ++index){
        params.source_indices_h[index] = src_indices[index];
        // std::cout << params.source_indices_h[index] << ", ";
    }
    
    std::cout << std::endl;
    
    std::cout << "dest_offsets: " ;
    for (int index = 0; index < dest_offsets.size(); ++index){
        params.destination_offsets_h[index] = dest_offsets[index];
        // std::cout << params.destination_offsets_h[index] << ", ";
    }
    std::cout << std::endl;

    std::cout << "weights_h: " ;
    for (int index = 0; index < weights.size(); ++index){
        params.weights_h[index] = weights[index];
        // std::cout << params.weights_h[index] << ", ";
    }
    std::cout << std::endl;
    // std::cout << std::endl;

    // std::cout << "here3" << std::endl;
    // std::cout << seeds_count << std::endl;
    // for(int i = 0; i < seeds_count; i++){
    //     params.source_indices_h[count] = seeds[i];
    //     params.weights_h[count] = 0;
    //     count++;
    // }
    // std::cout << "here4" << std::endl;

    return params;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);
    imagem *img = read_pgm(path);
    
    int n_fg, n_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;
    
    std::cout << img->cols << std::endl;
    std::cout << img->rows << std::endl;
    std::cout << img->total_size << std::endl;
    std::cout << (img->total_size * 4) - ((img->cols + img->rows)*2) << std::endl;
    // assert(n_fg == 1);
    // assert(n_bg == 1);
    

    //leitura de multiplos inputs
    std::vector<int> seeds_fg;
    std::vector<int> seeds_bg;
    int seed_bg, seed_fg;

    std::cout << n_fg << std::endl;
    std::cout << n_bg << std::endl;
    std::cout << std::endl;

    for(int i = 0; i < n_fg; i++){
        std::cin >> x >> y;
        std::cout << x << std::endl;
        std::cout << y << std::endl;
        std::cout << img->cols << std::endl;
        std::cout << std::endl;
        seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
    }

    for(int i = 0; i < n_bg; i++){
        std::cin >> x >> y;
        std::cout << x << std::endl;
        std::cout << y << std::endl;
        std::cout << img->cols << std::endl;
        std::cout << std::endl;
        seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
    }
        std::cout << "done 1" << std::endl;
    
    graphParams fg_params = GetGraphParams(img, seeds_fg, n_fg);
        std::cout << "c" << std::endl;
    std::cout << fg_params.n << std::endl;
    std::cout << fg_params.nnz << std::endl;
    graphParams bg_params = GetGraphParams(img, seeds_bg, n_bg);
        std::cout << "d" << std::endl;
    std::cout << bg_params.n << std::endl;
    std::cout << bg_params.nnz << std::endl;
    float * sssp_fg = (float*)malloc(fg_params.n*sizeof(float));
    NvidiaSSSH(fg_params.weights_h, fg_params.destination_offsets_h, fg_params.source_indices_h, fg_params.n, fg_params.nnz, seed_fg, sssp_fg);
    printf("sssp_fg\n");
    // for (int i = 0; i<fg_params.n; i++)  printf("%f\n",sssp_fg[i]); printf("\n");
    printf("\nDone!\n");

    float * sssp_bg = (float*)malloc(bg_params.n*sizeof(float));
    NvidiaSSSH(bg_params.weights_h, bg_params.destination_offsets_h, bg_params.source_indices_h, bg_params.n, bg_params.nnz, seed_bg, sssp_bg);
    printf("sssp_bg\n");
    // for (int i = 0; i<bg_params.n; i++)  printf("%f\n",sssp_bg[i]); printf("\n");
    printf("\nDone!\n");



    imagem *saida = new_image(img->rows, img->cols);
    
    for (int k = 0; k < saida->total_size; k++) {
        if (sssp_fg[k] > sssp_bg[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }
    
    write_pgm(saida, path_output);    
    return 0;
}
