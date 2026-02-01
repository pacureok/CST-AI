#ifndef TENTPU_ENGINE_H
#define TENTPU_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Procesa un tensor de video directamente en memoria.
 * * @param data Puntero al buffer de datos (float32).
 * @param size Número total de elementos en el tensor.
 */
void process_video_tensor(float* data, int size);

/**
 * @brief Devuelve la versión actual del motor PCURE-AI+.
 * * @return const char* Cadena de texto con la versión y optimización.
 */
const char* get_engine_version();

/**
 * @brief Inicializa los registros de hardware para TPU/ASM.
 * * @return int 0 si el éxito, 1 si hay error de hardware.
 */
int init_native_hardware();

#ifdef __cplusplus
}
#endif

#endif // TENTPU_ENGINE_H