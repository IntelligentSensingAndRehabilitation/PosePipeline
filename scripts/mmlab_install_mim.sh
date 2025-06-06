#!/bin/bash

sudo apt-get update && sudo apt-get install -y build-essential
pip install psutil ninja

mim install mmengine
mim install mmcv==2.1.0
mim install mmpretrain
mim install mmdet==3.2.0
mim install mmpose==1.3.2