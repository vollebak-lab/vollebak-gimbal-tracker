cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EVPropNet_Ingestion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Find Metavision SDK (OpenEB) & OpenCV
find_package(MetavisionSDK COMPONENTS stream core REQUIRED)
find_package(OpenCV REQUIRED)
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

# 7. Build Real-Time Propeller Flicker Detector Engine (with Ego-Motion Suppression & TensorRT FP16)
add_executable(ev_flicker_detector ev_flicker_detector.cpp)
target_link_libraries(ev_flicker_detector PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    ${OpenCV_LIBS}
    nvinfer
    cudart
    pthread
)

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
