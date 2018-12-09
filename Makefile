all: nvgraph.cu imagem.cpp
	nvcc -O2 -o nvgraph nvgraph.cu imagem.cpp --std=c++11 -lnvgraph; nvcc -O2 -o sequencial sequencial.cu imagem.cpp --std=c++11 

clean: 
	$(RM) nvgraph; $(RM) sequencial
