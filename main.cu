#include <iostream>
#include <queue>
#include <vector>
#include <assert.h>
#include <fstream>
#include <cuda_runtime.h>
#include "nvgraph.h"
#include "imagem.h"

typedef std::pair<double, int> custo_caminho;

typedef std::pair<double *, int *> result_sssp;

typedef std::pair<int, int> seed;

struct graphParams {
    float weights_h[];
    int destination_offsets_h[];
    int source_indices_h[];
    size_t n;
    size_t nnz;
};


void check(nvgraphStatus_t status) {
    if (status != NVGRAPH_STATUS_SUCCESS) {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

int NvidiaSSSH(float weights_h[], int destination_offsets_h[], int source_indices_h[], const size_t n, const size_t nnz) {
    const size_t vertex_numsets = 1, edge_numsets = 1;
    float *sssp_1_h;
    void** vertex_dim;
    // nvgraph variables
    nvgraphStatus_t status; nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    // Init host data
    sssp_1_h = (float*)malloc(n*sizeof(float));
    vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    vertex_dim[0]= (void*)sssp_1_h; vertex_dimT[0] = CUDA_R_32F;
    check(nvgraphCreate(&handle));
    check(nvgraphCreateGraphDescr (handle, &graph));
    CSC_input->nvertices = n; CSC_input->nedges = nnz;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    // Set graph connectivity and properties (tranfers)
    check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));
    // Solve
    int source_vert = 0;
    check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
    // Get and print result
    check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
    
    printf("sssp_1_h\n");
    for (int i = 0; i < n; i++)  printf("%f\n",sssp_1_h[i]); printf("\n");
    printf("\nDone!\n");

    //Clean 
    free(sssp_1_h); free(vertex_dim);
    free(vertex_dimT); free(CSC_input);
    check(nvgraphDestroyGraphDescr(handle, graph));
    check(nvgraphDestroy(handle));
    return 0;
}

struct compare_custo_caminho {
    bool operator()(custo_caminho &c1, custo_caminho &c2) {
        return c2.first < c1.first;
    }
};

graphParams GetGraphParams(imagem *img, std::vector<int> seeds, int seeds_count){

    graphParams params = {
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
    };
    params.nnz = img->total_size + 1; // +1 because of the mask seed to unify fg/bg
    params.n = img->total_size - (((img->cols + img->rows) * 2) - 8) + seeds_count;
    params.weights_h = new float[params.n]();
    params.destination_offsets_h = new int[params.nnz + 1](); // +1 to add the size as the last item to the end of the array
    params.source_indices_h = new int[params.n]();

    int count = 0;
    params.destination_offsets_h[0] = 0;
    for(int vertex = 0; vertex < params.nnz - 1; vertex++ ){

        int local_count = 0;

        int acima = vertex - img->cols;
        if (acima >= 0) {
            double custo_acima = get_edge(img, vertex, acima);
            params.source_indices_h[count] = acima;
            params.weights_h[count] = custo_acima;
            local_count++;
            count++;
        }

        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) {
            double custo_abaixo = get_edge(img, vertex, abaixo);
            params.source_indices_h[count] = abaixo;
            params.weights_h[count] = custo_abaixo;
            local_count++;
            count++;
        }


        int direita = vertex + 1;
        if (direita < img->total_size) {
            double custo_direita = get_edge(img, vertex, direita);
            params.source_indices_h[count] = direita;
            params.weights_h[count] = custo_direita;
            local_count++;
            count++;
        }

        int esquerda = vertex - 1;
        if (esquerda >= 0) {
            double custo_esquerda = get_edge(img, vertex, esquerda);
            params.source_indices_h[count] = esquerda;
            params.weights_h[count] = custo_esquerda;
            local_count++;
            count++;
        }

        params.destination_offsets_h[vertex + 1] = params.destination_offsets_h[vertex] + local_count;
    }

    //mask seed to all fg/bg

    params.destination_offsets_h[params.nnz - 2] = params.destination_offsets_h[params.nnz - 3] + 2;//because of the size of the last item
    params.destination_offsets_h[params.nnz - 1] = params.nnz;

    for(int i = 0; i < seeds_count; i++){
        params.source_indices_h[count] = seeds[i];
        params.weights_h[count] = 0;
        count++;
    }

    return params;
}

result_sssp SSSP(imagem *img, int source) {
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

    Q.push(custo_caminho(0.0, source));
    predecessor[source] = source;
    custos[source] = 0.0;

    while (!Q.empty()) {
        custo_caminho cm = Q.top();
        Q.pop();

        int vertex = cm.second;
        if (analisado[vertex]) continue; // já tem custo mínimo calculado
        analisado[vertex] = true;
        double custo_atual = cm.first;
        assert(custo_atual == custos[vertex]);

        int acima = vertex - img->cols;
        if (acima >= 0) {
            double custo_acima = custo_atual + get_edge(img, vertex, acima);
            if (custo_acima < custos[acima]) {
                custos[acima] = custo_acima;
                Q.push(custo_caminho(custo_acima, acima));
                predecessor[acima] = vertex;
            }
        }

        int abaixo = vertex + img->cols;
        if (abaixo < img->total_size) {
            double custo_abaixo = custo_atual + get_edge(img, vertex, abaixo);
            if (custo_abaixo < custos[abaixo]) {
                custos[abaixo] = custo_abaixo;
                Q.push(custo_caminho(custo_abaixo, abaixo));
                predecessor[abaixo] = vertex;
            }
        }


        int direita = vertex + 1;
        if (direita < img->total_size) {
            double custo_direita = custo_atual + get_edge(img, vertex, direita);
            if (custo_direita < custos[direita]) {
                custos[direita] = custo_direita;
                Q.push(custo_caminho(custo_direita, direita));
                predecessor[direita] = vertex;
            }
        }

        int esquerda = vertex - 1;
        if (esquerda >= 0) {
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
    imagem *img = read_pgm(path);
    
    int n_fg, n_bg;
    int x, y;
    
    std::cin >> n_fg >> n_bg;
    assert(n_fg == 1);
    assert(n_bg == 1);
    

    //leitura de multiplos inputs
    std::vector<int> seeds_fg;
    int seeds_fg_count;
    std::vector<int> seeds_bg;
    int seeds_bg_count;
    
    std::cin >> n_fg >> n_bg;

    for(int i = 0; i <= n_fg; i++){
        std::cin >> x >> y;
        int seed_fg = y * img->cols + x;
        seeds_fg.push_back(seed_fg);
        seeds_fg_count++;
    }

    for(int i = 0; i <= n_bg; i++){
        std::cin >> x >> y;
        int seed_bg = y * img->cols + x;
        seeds_bg.push_back(seed_bg);
        seeds_bg_count++;
    }
    
    graphParams fg_params = GetGraphParams(img, seeds_fg, seeds_fg_count);
    graphParams bg_params = GetGraphParams(img, seeds_bg, seeds_bg_count);
    
    NvidiaSSSH(fg_params.weights_h, fg_params.destination_offsets_h, fg_params.source_indices_h, fg_params.n, fg_params.nnz);

    // imagem *saida = new_image(img->rows, img->cols);
    
    // for (int k = 0; k < saida->total_size; k++) {
    //     if (fg.first[k] > bg.first[k]) {
    //         saida->pixels[k] = 0;
    //     } else {
    //         saida->pixels[k] = 255;
    //     }
    // }
    
    // write_pgm(saida, path_output);    
    return 0;
}
