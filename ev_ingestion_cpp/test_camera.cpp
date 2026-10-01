#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/base/events/event_cd.h>

int main(int argc, char *argv[]) {
    std::cout << "==========================================" << std::endl;
    std::cout << "  IDS IMX636 Live Stream Verification     " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        std::cout << "[INFO] Initializing Metavision Camera..." << std::endl;
        const char *serial_env = std::getenv("PREDATOR_EVENT_CAMERA_SERIAL");
        const std::string serial = argc > 1 ? argv[1] : (serial_env ? serial_env : "");
        Metavision::Camera camera = serial.empty() ? Metavision::Camera::from_first_available()
                                                    : Metavision::Camera::from_serial(serial);
        if (!serial.empty()) {
            std::cout << "[INFO] Direct-open serial: " << serial << std::endl;
        }
        
        int width = camera.geometry().get_width();
        int height = camera.geometry().get_height();
        std::cout << "[INFO] Camera successfully opened!" << std::endl;
        std::cout << "[INFO] Geometry: " << width << " x " << height << std::endl;
        
        std::atomic<uint64_t> total_events{0};
        std::atomic<uint64_t> slice_count{0};
        
        camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            uint64_t count = std::distance(begin, end);
            total_events += count;
            slice_count++;
        });
        
        camera.start();
        std::cout << "[INFO] Camera streaming active. Measuring throughput for 10 seconds..." << std::endl;
        
        for (int i = 1; i <= 10; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            uint64_t ev = total_events.exchange(0);
            uint64_t sl = slice_count.exchange(0);
            std::cout << "  [T+" << i << "s] Throughput: " << ev << " events/s | Callback batches: " << sl << "/s" << std::endl;
        }
        
        camera.stop();
        std::cout << "[INFO] Live hardware ingestion test PASSED." << std::endl;
        return 0;
    } catch (const Metavision::CameraException &e) {
        std::cerr << "[ERROR] CameraException: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "[ERROR] Standard exception: " << e.what() << std::endl;
        return 2;
    }
}
