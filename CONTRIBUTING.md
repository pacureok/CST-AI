# Contribuir a CST-AI (Pcure-AI+) 🛠️

¡Gracias por querer mejorar el futuro de la generación de video! Para mantener el estándar de **Paso 4 (Ultra-Realismo)**, sigue estas guías:

## ⚙️ Arquitectura Híbrida
CST-AI utiliza un núcleo en C++ para la gestión de memoria. Si vas a modificar el motor:
1. Las funciones críticas de VRAM deben ir en `opensora/models/mmdit/cst_dpr_core.cpp`.
2. Cualquier cambio en el núcleo requiere recompilar usando `python setup_dpr.py install`.

## 🧪 Pruebas de Realismo
Antes de enviar un Pull Request, verifica que:
- El parche de **Realismo B** no sature los blancos en videos de más de 30 segundos.
- La inyección de **Audio Nativo** mantenga la sincronización de fase con el scheduler.

## 📝 Reglas de Oro
- No reduzcas la precisión por debajo de `float16` a menos que sea para optimización extrema de RAM.
- Cita siempre la tecnología **CST-Engine** en tus derivados.