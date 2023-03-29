INPUT_SHAPES="1,1,640,640"

OPS="nn.conv2d,nn.batch_norm,nn.relu,nn.bias_add,nn.max_pool2d,nn.avg_pool2d"
DEVICE_KEY="thelios"
MODEL_NAME="faster-rcnn-end2end"
MODEL_PATH=/home/ian/_/tvm-tuning-light/models/faster-rcnn-end2end.onnx
TARGET="cuda"
TUNING_MODE="autoscheduler"

# Check user wishes to continue after reviewing the parameters
echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Input shapes: $INPUT_SHAPES"
echo "Device key: $DEVICE_KEY"
echo "Ops: $OPS"
echo "Target: $TARGET"

# Check with user before continuing
read -p "Continue? (y/n) " -n 1 -r
python tuner.py --model-name=$MODEL_NAME --model-path=$MODEL_PATH --device-key=$DEVICE_KEY --target=$TARGET --tuning-mode=$TUNING_MODE
