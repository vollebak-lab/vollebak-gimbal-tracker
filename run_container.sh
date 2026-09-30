#!/bin/bash
pip install opencv-python-headless scipy imutils tqdm matplotlib scikit-image "numpy<=1.24.3" "protobuf<4" tf2onnx
cd /workspace/EVPropNet/Code
sed -i 's/name=" NetworkOutput)/name="NetworkOutput")/g' Train.py
echo "Generating data..."
mkdir -p /workspace/datasets/evpropnet/Imgs
python3 GenData.py --BgImgPath /workspace/datasets/coco/train2014/ --WritePath /workspace/datasets/evpropnet/ --NumImages 1000
echo "Training model..."
mkdir -p /workspace/checkpoints /workspace/logs
python3 Train.py --BasePath /workspace/datasets/evpropnet/ --CheckPointPath /workspace/checkpoints/ --LogsPath /workspace/logs/ --NumEpochs 5
echo "Exporting to ONNX..."
LATEST_CKPT=$(ls -t /workspace/checkpoints/*.meta | head -n 1 | sed 's/\.meta//')
python3 -m tf2onnx.convert --checkpoint "$LATEST_CKPT" --output /workspace/evpropnet.onnx --inputs Input:0 --outputs NetworkOutput:0
