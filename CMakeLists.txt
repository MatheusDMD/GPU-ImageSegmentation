cmake_minimum_required(VERSION 2.6)
project(segmentacao_sequencial)

add_executable(segmentacao_sequencial main.cpp imagem.cpp)

install(TARGETS segmentacao_sequencial RUNTIME DESTINATION bin)
