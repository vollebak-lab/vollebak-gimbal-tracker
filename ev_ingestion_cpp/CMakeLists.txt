cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EVPropNet_Ingestion CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87) # Jetson Orin Nano (Ampere sm_87)

# 1. Find Metavision SDK (OpenEB) & OpenCV
find_package(MetavisionSDK COMPONENTS stream core REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
message(STATUS "Found Metavision SDK: ${MetavisionSDK_FOUND}")
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# 2. CUDA & TensorRT Configuration
include_directories(
    /usr/local/cuda/include
    /usr/include/aarch64-linux-gnu
)
link_directories(
    /usr/local/cuda/lib64
    /usr/lib/aarch64-linux-gnu
)

# 3. Build Hardware Stream Test Utility
add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    pthread
)

# 4. Build Live Neuromorphic Web Streamer UI
add_executable(ev_web_viewer ev_web_viewer.cpp)
target_link_libraries(ev_web_viewer PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    ${OpenCV_LIBS}
    pthread
)

# 5. Build Frequency-Domain DSP Unit Test Suite
add_executable(test_flicker_dsp test_flicker_dsp.cpp)
target_link_libraries(test_flicker_dsp PRIVATE pthread)

# 6. Build Ego-Motion Compensation & Gyro Warping Test Suite
add_executable(test_ego_motion test_ego_motion.cpp)
target_link_libraries(test_ego_motion PRIVATE pthread)

# 6b. Build Live Nicla Sense ME Stream Verifier
add_executable(test_nicla_live test_nicla_live.cpp)
target_link_libraries(test_nicla_live PRIVATE pthread)

# 6c. Build CUDA & cuFFT Acceleration Test Suite
add_executable(test_cuda_flicker test_cuda_flicker.cpp cuda_flicker_core.cu)
target_link_libraries(test_cuda_flicker PRIVATE 
    MetavisionSDK::stream
    MetavisionSDK::core
    nvinfer
    cufft
    cudart
    pthread
)

# Test suites rely on assert(). Release adds -DNDEBUG globally, which silently compiled every
# assertion away (Phase 33.5 finding); keep -O3 but re-enable assertions for the test targets.
foreach(test_target test_flicker_dsp test_ego_motion test_cuda_flicker)
    target_compile_options(${test_target} PRIVATE -UNDEBUG)
endforeach()

# 7. Build Real-Time Propeller Flicker Detector Engine (with Ego-Motion Suppression & TensorRT FP16)
add_executable(ev_flicker_detector ev_flicker_detector.cpp cuda_flicker_core.cu)
target_link_libraries(ev_flicker_detector PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    nvinfer
    cufft
    cudart
    pthread
)
# Build provenance: deploy script passes -DPREDATOR_BUILD_ID=<git>-<src sha256 prefix>
set(PREDATOR_BUILD_ID "unstamped" CACHE STRING "Build provenance stamp (git rev + source hash)")
target_compile_definitions(ev_flicker_detector PRIVATE PREDATOR_BUILD_ID="${PREDATOR_BUILD_ID}")

# 7b. Phase 33.4b ingest microbenchmark (opt-in: `cmake --build . --target bench_ingest`).
# Times OpenEB EVT3 decode, frame generation and ingest_event_batch per event, after verifying the
# reference EVT3 encoder round-trips bit-exactly through OpenEB's decoder.
find_package(MetavisionHAL REQUIRED)
add_executable(bench_ingest EXCLUDE_FROM_ALL bench_ingest.cpp cuda_flicker_core.cu)
target_link_libraries(bench_ingest PRIVATE
    Metavision::HAL
    MetavisionSDK::core
    ${OpenCV_LIBS}
    cufft
    cudart
    pthread
)

# 7c. Phase 33.4b EVT2.1 live capture + layout verification (opt-in). Writes raw + OpenEB-decoded
# fixtures used by the GPU decoder parity test. Requires exclusive camera access (stop the service).
add_executable(evt21_capture EXCLUDE_FROM_ALL evt21_capture.cpp)
target_link_libraries(evt21_capture PRIVATE Metavision::HAL Metavision::HAL_discovery MetavisionSDK::core pthread)

# 7d. Phase 33.4b.b GPU EVT2.1 decoder parity test against OpenEB's EVT21LegacyDecoder. The synthetic suite
# runs in the deploy script; `--fixture <prefix>` replays an evt21_capture recording and reports timing.
add_executable(test_evt21_decoder test_evt21_decoder.cpp evt21_gpu_decoder.cu)
target_link_libraries(test_evt21_decoder PRIVATE Metavision::HAL MetavisionSDK::core cudart pthread)
target_compile_options(test_evt21_decoder PRIVATE -UNDEBUG)

# 7e. Phase 33.7a Hot pixel mask unit tests and survey tool (IMX636 I_DigitalEventMask)
add_executable(test_hot_pixel_mask test_hot_pixel_mask.cpp)
target_link_libraries(test_hot_pixel_mask PRIVATE Metavision::HAL MetavisionSDK::core pthread)
target_compile_options(test_hot_pixel_mask PRIVATE -UNDEBUG)

add_executable(hot_pixel_survey EXCLUDE_FROM_ALL hot_pixel_survey.cpp)
target_link_libraries(hot_pixel_survey PRIVATE Metavision::HAL Metavision::HAL_discovery MetavisionSDK::core pthread)

# 7f. Phase 33.4b.d GPU Micro-Neighborhood Periodicity Sieve parity test & benchmark
add_executable(test_gpu_sieve test_gpu_sieve.cpp gpu_sieve.cu)
target_link_libraries(test_gpu_sieve PRIVATE cudart pthread)
target_compile_options(test_gpu_sieve PRIVATE -UNDEBUG)

# 8. Optional ONNX Runtime Ingestion Binary
find_package(onnxruntime QUIET)
if (onnxruntime_FOUND OR EXISTS "/usr/include/onnxruntime" OR EXISTS "/usr/local/include/onnxruntime")
    add_executable(ev_ingestion_onnx main_onnx.cpp)
    target_link_libraries(ev_ingestion_onnx PRIVATE 
        MetavisionSDK::stream 
        MetavisionSDK::core 
        onnxruntime
        pthread
    )
endif()
