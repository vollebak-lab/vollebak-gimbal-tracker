#!/bin/bash
set -e
cd /root/evpropnet_workspace
mkdir -p datasets/coco
cd datasets/coco
if [ ! -d "train2014" ]; then
  echo "Downloading MS COCO backgrounds..."
  wget -qO val2014.zip http://images.cocodataset.org/zips/val2014.zip
  unzip -q val2014.zip
  mv val2014 train2014
  rm val2014.zip
fi
echo "Backgrounds ready."
