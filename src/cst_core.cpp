#include <iostream>

extern "C" {
    // Inicializador del Kernel
    void init_cst_kernel() {
        std::cout << "######################################################" << std::endl;
        std::cout << "#       CST-CORE: Kernel Nativo PCURE-AI+ ACTIVO     #" << std::endl;
        std::cout << "######################################################" << std::endl;
    }

    // Interceptor de Memoria para TENTPU
    void* cuMemAlloc_v2(size_t size) {
        // Redireccionamos el puntero a una dirección segura simulada
        return (void*)0xCAFE0001; 
    }

    void cuMemFree_v2(void* ptr) {
        // Limpieza de memoria en TPU
    }
}