MODEL_NAME=mnist-1
INPUT_SHAPES="1,1,640,640"
MODEL_PATH=/home/ubuntu/tvm-tuning-light/models/mnist-1.onnx
OPS="nn.conv2d,nn.dense,nn.batch_flatten,nn.relu,nn.max_pool2d,nn.avg_pool2d,nn.global_avg_pool2d,nn.batch_norm,nn.softmax"
DEVICE_KEY="raspi"
#MODEL_NAME=resnet50
#MODEL_PATH=/home/ubuntu/tvm-tuning-light/models/resnet50-v2-7.onnx

MODEL_NAME="faster-rcnn-end2end"
MODEL_PATH=/home/ubuntu/tvm-tuning-light/models/faster-rcnn-end2end.onnx

# Check user wishes to continue after reviewing the parameters
echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Input shapes: $INPUT_SHAPES"
echo "Device key: $DEVICE_KEY"
echo "Ops: $OPS"

# Check with user before continuing
read -p "Continue? (y/n) " -n 1 -r
python tuner.py --model-name=$MODEL_NAME --model-path=$MODEL_PATH --device-key=$DEVICE_KEY --ops=$OPS
