#include <iostream>
#include <vector>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <onnxruntime_cxx_api.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_evpropnet_model.onnx>\n";
        return 1;
    }

    // 1. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EVPropNet");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    
    // Create ONNX session
    Ort::Session session(env, argv[1], session_options);
    
    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    std::cout << "ONNX Model loaded successfully.\n";

    // 2. Initialize the IDS Camera via Metavision SDK
    Metavision::Camera camera;
    try {
        camera = Metavision::Camera::from_first_available();
    } catch (const Metavision::CameraException& e) {
        std::cerr << "Camera Error: " << e.what() << std::endl;
        return -1;
    }

    int width = camera.geometry().get_width();
    int height = camera.geometry().get_height();

    // 3. Set up the Slice Accumulator (10ms)
    const uint32_t slice_duration_us = 10000; 
    std::vector<Metavision::EventCD> current_slice;

    // 4. The Event Callback
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (auto it = begin; it != end; ++it) {
            current_slice.push_back(*it);
            
            if (current_slice.back().t - current_slice.front().t >= slice_duration_us) {
                // --- A. Convert Events to 2D Tensor (NHWC for TF) ---
                // EVPropNet (TF) expects (Batch, Height, Width, Channels)
                std::vector<int64_t> input_shape = {1, height, width, 2};
                size_t tensor_size = height * width * 2;
                std::vector<float> tensor_data(tensor_size, 0.0f);

                for (const auto& ev : current_slice) {
                    // Indexing: Y * W * C + X * C + Channel
                    int channel = ev.p ? 1 : 0;
                    int index = ev.y * width * 2 + ev.x * 2 + channel;
                    tensor_data[index] += 1.0f;
                }

                // --- B. Inference ---
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info, tensor_data.data(), tensor_size, input_shape.data(), input_shape.size());

                auto output_tensors = session.Run(
                    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
                
                // (Add post-processing here)
                
                current_slice.clear();
            }
        }
    });

    camera.start();
    std::cout << "Camera streaming... Press Enter to stop." << std::endl;
    std::cin.get();

    camera.stop();
    return 0;
}
