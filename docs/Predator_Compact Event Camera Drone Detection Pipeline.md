# Predator: Compact Event Camera Drone Detection Pipeline

This document outlines the technical specification and implementation plan for building the event camera data processing software pipeline to detect, classify, and track long-range drones using a Double-Helix Point Spread Function (DH-PSF) phase mask. 

The build is designed for NVIDIA DGX Spark (training) and will transition to the NVIDIA Jetson Orin NX 16GB (edge deployment).

## Decisions & Resolves

> [!NOTE]
> **Data Strategy:**
> - **Classifier Data:** We will download and use the **EVPropNet** dataset directly. Because the 1D CNN classifier analyzes the temporal micro-Doppler frequency (Blade Passage Frequency) rather than spatial shape, it does not require DH-PSF blur application.
> - **Tracking/Geometric Data:** We will download the **FRED** dataset and programmatically apply a synthetic DH-PSF blur. This synthetic data will serve as the baseline for the Coarse Clustering and Fine Dual-Lobe Separation algorithms.
> - **Lens Selection:** We will use the **Vision Datum 8mm M12** (or equivalent 8mm M12 board lens) as the baseline for generating the synthetic DH-PSF blur parameters. M12 lenses are optimal for the direct-on-glass DH-PSF printing and edge SWaP constraints.
>
> **Comms Protocol / IPC:**
> - We will adopt **Zenoh** as the communication backbone. Zenoh provides extremely low latency, high throughput, and supports zero-copy communication. It is highly efficient for edge compute environments (like the Orin NX), natively supports both C++ and Python/Rust, and aligns with your existing technology stack. 
> - **Input Format:** Since live sensor data is not yet available, the ingestion engine will read standard event data formats (e.g., `.raw`, `.aedat4`, or `hdf5`) produced by our synthetic pipelines.

---

## Proposed Pipeline Architecture

Based on the Scheimpflug principles and DH-PSF engineering research, the pipeline requires two independent branches gated by classification to conserve compute.

```mermaid
graph TD
    A[Raw Asynchronous Event Stream (.aedat4 / Zenoh Pub)] --> B(Spatiotemporal Noise Filter)
    B --> C(Coarse Spatial Clustering)
    C -->|Centroid Az/El + Target ROI via Zenoh| D{YES/NO Gate: Classifier}
    
    subgraph Track B: Classification
    D_Pre[Crop Events to ROI] --> D_Spec[Spectral Extraction: 1D FFT]
    D_Spec --> D_CNN[1D CNN / TensorRT]
    D_CNN -->|Threat Classification via Zenoh| D
    end
    
    C --> D_Pre
    
    subgraph Track A: Ranging & Kinematics
    D -->|YES: Confirmed Target| E(Fine Dual-Lobe Separation k=2)
    E --> F(Rotation Angle Calculation θ)
    F --> G(Depth Mapping z)
    G --> H(Asynchronous Kalman Filter)
    end
    
    C -.->|Target Centroid| H
    H --> I[3D Kinematic Track: x, y, z, velocity]
```

---

## Implementation Phases

### Phase 0: Data Acquisition & Synthetic Generation
**Language:** Python
**Focus:** Establishing the baseline data required for all downstream phases.
* **EVPropNet Pipeline:** Download and parse EVPropNet datasets for the classifier.
* **FRED Pipeline:** Download the FRED dataset. 
* **DH-PSF Simulator:** Build a Python simulator to apply the theoretical optical blur of the DH-PSF (modeled on an 8mm M12 lens) to the FRED event streams, generating `.aedat4` or `.h5` files with dual-lobe characteristics.

### Phase 1: Core Framework & Front-End Filtering
**Language:** C++ / CUDA (with Zenoh)
**Focus:** Handling the raw microsecond event stream and filtering noise.
* Build the base event ingestion engine to read from synthetic files and publish via Zenoh.
* Implement the Spatiotemporal Noise Filter using a Decaying Time Surface (Surface of Active Events) to isolate high-frequency rotor flicker.
* Implement Coarse Spatial Clustering (DBSCAN or Density Filter) to generate bounding boxes (ROIs) and Azimuth/Elevation centroids. Publish ROIs to Zenoh.

### Phase 2: Classification Branch (Micro-Doppler)
**Language:** Python (Training on DGX) / C++ TensorRT (Edge Inference)
**Focus:** Determining what the object is based on its frequency signature.
* Subscribe to ROI via Zenoh and crop events.
* Implement Spectral Extraction using a CUDA-accelerated Non-Uniform Discrete Fourier Transform (NDFT) or sliding-window 1D FFT on the event rate.
* Design and train a lightweight 1D CNN using EVPropNet to classify harmonic signatures.
* Compile the classifier to TensorRT for edge inference and publish classification state to Zenoh.

### Phase 3: The Conditional Execution Gate & Tracking Branch (Geometric Ranging)
**Language:** C++ (Eigen)
**Focus:** Gated execution and calculating the 3D position of confirmed targets.
* Implement the YES/NO logic gate listening to the Zenoh classification topic. 
* Implement Fine Dual-Lobe Separation using a 2-component Gaussian Mixture Model (GMM) restricted to the active ROI on the DH-PSF FRED data.
* Implement Rotation Angle Calculation ($\theta = \text{arctan2}(y_2 - y_1, x_2 - x_1)$) ensuring phase unwrapping over time.
* Implement Depth Mapping using an empirical polynomial calibration function ($z = f(\theta)$).
* Build the Asynchronous Extended Kalman Filter (EKF) to ingest depth ($z$) and front-end centroid (Az/El) to smooth the 3D state vector.

### Phase 4: Hardware Optimization & Deployment
* End-to-end integration of all phases across the Zenoh network.
* Profiling memory bandwidth and tensor core utilization on the target NVIDIA Jetson Orin NX hardware.

---

## Verification Plan

### Automated Tests
- `test_noise_filter`: Inject synthetic noise events and verify the Time Surface correctly drops them.
- `test_clustering`: Provide a simulated dual-lobe event cluster and verify bounding box accuracy.
- `test_rotation_math`: Feed synthetic $(x_1, y_1), (x_2, y_2)$ coordinates to verify $\theta$ calculation and un-wrapping logic.

### Manual Verification
- Deploy the pipeline on the DGX Spark using pre-recorded event camera datasets (e.g., EVPropNet or FRED datasets with synthetically applied DH-PSF).
- Visually verify the target classification and 3D bounding box tracking via a custom visualization tool before transitioning to the Orin NX.
