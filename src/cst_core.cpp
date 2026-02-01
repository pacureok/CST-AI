#include <iostream>
extern "C" {
    void init_cst_kernel() {
        std::cout << "[💎] CST-CORE: Motor C++ Nativo ejecutándose en Google TPU." << std::endl;
    }
    float sync_audio_video_latence(float audio_ts, float video_ts) {
        return (audio_ts > video_ts) ? (audio_ts - video_ts) : (video_ts - audio_ts);
    }
}
