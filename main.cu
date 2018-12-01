#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <algorithm>
#include <iterator> 
#include <cuda_runtime.h>
#include <nvgraph.h>
#include "imagem.h"

typedef std::pair<double, int> cost_caminho;
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

int NvidiaSSSH(float *weights_h, int *destination_offsets_h, int *source_indices_h, const size_t n, const size_t nnz, int source_seed_bg, int source_seed_fg, float *sssp_1_h, float *sssp_2_h) {
    const size_t vertex_numsets = 2, edge_numsets = 1;
    void** vertex_dim;

    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;

    // Init host data
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h;
    vertex_dim[1]= (void*)sssp_2_h;
    vertex_dimT[0] = CUDA_R_32F;
    vertex_dimT[1] = CUDA_R_32F;

    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    
    // SOLVE BG
    int source_vert = source_seed_bg; //source_seed
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));

    // SOLVE FG
    source_vert = source_seed_fg; //source_seed
    check_status(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    check_status(nvgraphGetVertexData(handle, graph, (void*)sssp_2_h, 0));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);
    
    //Clean 
    check_status(nvgraphDestroyGraphDescr (handle, graph));
    check_status(nvgraphDestroy (handle));
    
    return 0;
}

graphParams GetGraphParams(imagem *img, std::vector<int> seeds_bg, std::vector<int> seeds_fg, int seeds_count){
    std::vector<int> dest_offsets;
    std::vector<int> src_indices;
    std::vector<float> weights;
    
    dest_offsets.push_back(0); // add zero to initial position

    //LOOP OVER ALL VERTEX
    for(int vertex = 0; vertex < img->total_size; vertex++ ){

        int local_count = 0;
        int vertex_i = vertex / img->cols;
        int vertex_j = vertex % img->cols;

        // CHECK IF THERE'S ITEM IN SPECIFIC DIRECTION AND APPENDS THE RESPECTIVE VALUES
        //ABOVE
        if (vertex_i > 0) {
            int above = vertex - img->cols;
            double cost_above = get_edge(img, vertex, above);
            src_indices.push_back(above);
            weights.push_back(cost_above);
            local_count++;
        }
        //BELOW
        if (vertex_i < img->rows - 1) {
            int below = vertex + img->cols;
            double cost_below = get_edge(img, vertex, below);
            src_indices.push_back(below);
            weights.push_back(cost_below);
            local_count++;
        }
        //RIGHT
        if (vertex_j < img->cols - 1) {
            int right = vertex + 1;
            double cost_right = get_edge(img, vertex, right);
            src_indices.push_back(right);
            weights.push_back(cost_right);
            local_count++;
        }
        //LEFT
        if (vertex_j > 0) {
            int left = vertex - 1;
            double cost_left = get_edge(img, vertex, left);
            src_indices.push_back(left);
            weights.push_back(cost_left);
            local_count++;
        }

        // CHECK IF THE CURRENT POSITION IS A SEED
        if (std::find(std::begin(seeds_bg), std::end(seeds_bg), vertex) != std::end(seeds_bg)){
            // ADDS THE VALUE OF THE LAST NODE TO THE SRC INDEX AND PUSHES A ZERO VALUE WEIGHT
            src_indices.push_back(img->total_size);
            weights.push_back(0.0);
            local_count++;
        }else if (std::find(std::begin(seeds_fg), std::end(seeds_fg), vertex) != std::end(seeds_fg)){
            src_indices.push_back(img->total_size + 1);
            weights.push_back(0.0);
            local_count++;
        }

        dest_offsets.push_back(dest_offsets.back() + local_count); // add local_count to last position vector
    }

    // ALOCATE ARRAYS AND STRUCT
    graphParams params = {};
    params.n = dest_offsets.size() - 1;
    params.nnz = ((img->total_size * 4) - ((img->cols + img->rows) * 2)) + seeds_count; //total connections (+ seed_count to add the connections from all nodes from that type)
    params.source_indices_h = (int*) malloc(params.nnz*sizeof(int));
    params.weights_h = (float*)malloc(params.nnz*sizeof(float));
    params.destination_offsets_h = (int*) malloc((params.n+1)*sizeof(int));
    
    // CONVERT STD:VECTORS IN ALOCATED ARRAYS
    for (int index = 0; index < src_indices.size(); ++index){
        params.source_indices_h[index] = src_indices[index];
    }
    for (int index = 0; index < dest_offsets.size(); ++index){
        params.destination_offsets_h[index] = dest_offsets[index];
    }
    for (int index = 0; index < weights.size(); ++index){
        params.weights_h[index] = weights[index];
    }
    return params;
}

int main(int argc, char **argv) {
    // CMD LINE ARGUMENTS
    if (argc < 3) {
        std::cout << "Uso:  segmentacao_sequencial entrada.pgm saida.pgm\n";
        return -1;
    }
    std::string path(argv[1]);
    std::string path_output(argv[2]);

    // READ IMAGE
    imagem *img = read_pgm(path);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    int n_fg, n_bg;
    int x, y;
    std::cin >> n_fg >> n_bg;
    
    // READ MULTIPLE SEEDS FROM INPUT FILE
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
    std::cerr << "INPUT - OK" << std::endl;
    
    // GET PARAMETERS TO NVGRAPH SSSP FUNCTION
    graphParams params = GetGraphParams(img, seeds_bg, seeds_fg, n_fg + n_bg);
    std::cerr << "PARAMS CREATION - OK" << std::endl;

    // ARRAYS TO STORE DISTANCE NODES
    float * sssp_fg = (float*)malloc(params.n*sizeof(float));
    float * sssp_bg = (float*)malloc(params.n*sizeof(float));

    // cudaEventRecord(start);
    // CALCULATE DISTANCE TO NODES
    NvidiaSSSH(params.weights_h, params.destination_offsets_h, params.source_indices_h, params.n, params.nnz, img->total_size, img->total_size+1, sssp_bg, sssp_fg);
    std::cerr << "DISTANCES CALCULATED - OK" << std::endl;
    // cudaEventRecord(stop);

    // float elapsed_time;
    // cudaEventElapsedTime(&elapsed_time, start, stop);
    // std::cerr << elapsed_time << std::endl;

    // OUTPUT IMAGE
    imagem *saida = new_image(img->rows, img->cols);

    // DISTANCE COMPARISON
    for (int k = 0; k < saida->total_size; k++) {
        // WHITE -> FOREGROUND
        // BLACK -> BACKGROUND
        if (sssp_fg[k] > sssp_bg[k]) {
            saida->pixels[k] = 0;
        } else {
            saida->pixels[k] = 255;
        }
    }

    // WRITE OUTPUT IMAGE
    write_pgm(saida, path_output);    
    std::cerr << "IMAGE OUTPUT - OK" << std::endl;


    // cudaDestroy(&start);
    // cudaDestroy(&stop);

    return 0;
}
