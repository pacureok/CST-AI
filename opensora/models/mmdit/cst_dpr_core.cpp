#include <torch/extension.h>
#include <vector>

// CST DPR: Función para mover tensores de forma asíncrona y liberar caché
void dpr_memory_swap(torch::Tensor tensor, std::string target_device) {
    if (target_device == "cpu") {
        tensor.to(torch::kCPU, /*non_blocking=*/true);
    } else {
        tensor.to(torch::kCUDA, /*non_blocking=*/true);
    }
}

// Limpiador agresivo de fragmentación de VRAM
void clear_vram_cache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dpr_swap", &dpr_memory_swap, "CST DPR Tensor Swapper");
    m.def("clear_vram", &clear_vram_cache, "CST Aggressive VRAM Cleaner");
}