#include <iostream>
#include <vector>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <torch/script.h> // LibTorch Header
#include <torch/torch.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_evpropnet_model.pt>\n";
        return 1;
    }

    // 1. Load the TorchScript EVPropNet model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(argv[1]);
        model.to(at::kCUDA); // Move model to Orin Nano's GPU
        model.eval();
        std::cout << "Model loaded successfully to CUDA.\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return -1;
    }

    // 2. Initialize the IDS Camera via Metavision SDK
    Metavision::Camera camera;
    try {
        camera = Metavision::Camera::from_first_available();
    } catch (const Metavision::CameraException& e) {
        std::cerr << "Camera Error: " << e.what() << std::endl;
        return -1;
    }

    // Define sensor geometry (1280x720 for the UE-39B0XCP-E)
    int width = camera.geometry().get_width();
    int height = camera.geometry().get_height();

    // 3. Set up the Slice Accumulator (e.g., 10ms windows = 10,000 us)
    const uint32_t slice_duration_us = 10000; 
    std::vector<Metavision::EventCD> current_slice;

    // 4. The Event Callback (Triggered asynchronously by the camera)
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (auto it = begin; it != end; ++it) {
            current_slice.push_back(*it);
            
            // Check if we have accumulated 10ms of events
            if (current_slice.back().t - current_slice.front().t >= slice_duration_us) {
                
                // --- A. Convert Events to 2D Tensor (Histogram) ---
                // EVPropNet typically expects a (2, Height, Width) tensor 
                // Channel 0: Negative events, Channel 1: Positive events
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                torch::Tensor event_tensor = torch::zeros({1, 2, height, width}, options);
                
                // Obtain a data pointer to populate on CPU
                float* tensor_data = event_tensor.data_ptr<float>();

                for (const auto& ev : current_slice) {
                    // Indexing: Batch(0) * (2*H*W) + Channel * (H*W) + Y * W + X
                    int channel_offset = (ev.p ? 1 : 0) * (height * width);
                    int pixel_offset = ev.y * width + ev.x;
                    if (pixel_offset >= 0 && pixel_offset < height * width) {
                        tensor_data[channel_offset + pixel_offset] += 1.0f;
                    }
                }

                // --- B. Inference on CUDA ---
                torch::Tensor cuda_tensor = event_tensor.to(torch::kCUDA);
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(cuda_tensor);

                // Run EVPropNet
                auto output = model.forward(inputs).toTensor();
                
                // (Add your post-processing logic here for bounding boxes)
                
                // Clear the slice for the next 10ms batch
                current_slice.clear();
            }
        }
    });

    // Start the camera
    camera.start();
    std::cout << "Camera streaming... Press Enter to stop." << std::endl;
    std::cin.get();

    camera.stop();
    return 0;
}
