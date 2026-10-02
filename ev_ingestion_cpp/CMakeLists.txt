cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(EVPropNet_Ingestion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TENSORRT_DETECTOR "Build the CUDA/TensorRT event detector (Jetson/NVIDIA only)" OFF)

# 1. Find Metavision SDK (OpenEB) & OpenCV
find_package(MetavisionSDK COMPONENTS stream core REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "Found Metavision SDK: ${MetavisionSDK_FOUND}")
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")

# 2. Build Hardware Stream Test Utility
add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    pthread
)

# 3. Build Live Neuromorphic Web Streamer UI
add_executable(ev_web_viewer ev_web_viewer.cpp)
target_link_libraries(ev_web_viewer PRIVATE 
    MetavisionSDK::stream 
    MetavisionSDK::core
    ${OpenCV_LIBS}
    pthread
)

# 4. Build Frequency-Domain DSP Unit Test Suite
add_executable(test_flicker_dsp test_flicker_dsp.cpp)
target_link_libraries(test_flicker_dsp PRIVATE pthread)

# 5. Build Ego-Motion Compensation & Gyro Warping Test Suite
add_executable(test_ego_motion test_ego_motion.cpp)
target_link_libraries(test_ego_motion PRIVATE pthread)

# 6. Build the real-time propeller flicker detector on every target. CUDA and
# TensorRT add dynamic event suppression when requested; the CPU build keeps
# Bart's spatial FFT detector and uses a pass-through suppression stage.
add_executable(ev_flicker_detector ev_flicker_detector.cpp)
target_link_libraries(ev_flicker_detector PRIVATE
    MetavisionSDK::stream
    MetavisionSDK::core
    ${OpenCV_LIBS}
    pthread
)

if (BUILD_TENSORRT_DETECTOR)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu)
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_package(CUDAToolkit REQUIRED)

    if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
        message(FATAL_ERROR "TensorRT was requested but NvInfer.h/libnvinfer were not found")
    endif()

    target_compile_definitions(ev_flicker_detector PRIVATE PREDATOR_WITH_TENSORRT=1)
    target_include_directories(ev_flicker_detector PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(ev_flicker_detector PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart
    )
endif()

# 7. Optional ONNX Runtime Ingestion Binary
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
