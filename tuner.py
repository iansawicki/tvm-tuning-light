import tvm 
import os
import argparse
import subprocess
from pathlib import Path
from tvm import relay, autotvm
import onnx
from datetime import datetime as dt
import platform
import time


# Check TVM version 
def check_tvm_version():
    print(tvm.__version__)
    
# Default TVM tuning options
num_measure_trials = 1000
opt_level = 3
verbose = True

# Default RPC options
rpc_runner = dict(host="127.0.0.1",
              port=9190,
              timeout=30,
              repeat=1,
              number=5,
              min_repeat_ms=200,
              enable_cpu_cache_flush=True)

# Default tuning options
early_stopping = 300
tuner =  tvm.autotvm.tuner.XGBTuner
num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)
tuner_settings = dict(loss_type="rank", feature_type="know")


# Create /models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")
    
# Create /tuning_logs directory if it doesn't exist
if not os.path.exists("tuning_logs"):
    os.makedirs("tuning_logs")
    
    
# Default path to models
MODELS = Path("models")
TUNING_LOGS = Path("tuning_logs")
    
def create_logging_file(model_name, tuning_method="autotvm"):
    
    # Build tuning log path name
    epoch_time = str(dt.now().timestamp()).split(".")[0]
    platform_arch = platform.machine()
    vendor_name = platform.node()
    logfile = TUNING_LOGS / f'{tuning_method}_{model_name}_{platform_arch}_{vendor_name}_{epoch_time}.log'
    return logfile

def extract_graph_tasks(onnx_model, target="llvm",*args, **kwargs):
    # Extract tasks from the graph
    mod, params = relay.frontend.from_onnx(onnx_model, shape=inputs, dtype={input_name: input_dtype},opset=11)
    print("Extract tasks...")
    if kwargs["ops"] is not None:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params, ops=kwargs["ops"])
    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    return tasks

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end-start} seconds to execute.")
        print("Total thread count: ", num_threads)
        return result
    return wrapper
    

@time_it
def tune_model_tasks(tasks, logfile_name, device_key , *args, **kwargs):
     # create tmp log file
    tmp_log_file = str(logfile_name) + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    
    # Tune tasks
    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        
        # Create builder, runner, and measure_option
        builder = autotvm.LocalBuilder()
        runner=autotvm.RPCRunner(device_key,**rpc_runner)
        measure_option = autotvm.measure_option(builder=builder, runner=runner)
        
        # Create tuner_obj
        if num_threads:
            tuner_settings["num_threads"] = num_threads
        tuner_obj = tuner(task, **tuner_settings)

        # Process_tuning
        tsk_trial = min(num_measure_trials, len(task.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, logfile_name)
    os.remove(tmp_log_file)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Check version
    parser.add_argument("--version", action="store_true", help="Check TVM version")
    # Download toy model
    parser.add_argument("--download-model", action="store_true", help="Download a toy model")
    # Set path to model
    parser.add_argument("--model-path", type=str, help="Path to model")
    # Set model name
    parser.add_argument("--model-name", type=str, help="Name of model")
    # Set batch_size, default is 1
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    # Set number of trials, default is 1000
    parser.add_argument("--trials", type=int, default=1000, help="Number of trials")
    # Set number of threads, default is 1
    parser.add_argument("--threads", type=int, default=1, help="Number of threads")
    # Set target device, default is "llvm"
    parser.add_argument("--target", type=str, default="llvm", help="Target device")
    # Set target host, default is "llvm"
    parser.add_argument("--target-host", type=str, default="llvm", help="Target host")
    # Set input size, default is 640x640
    parser.add_argument("--img-size", type=int, default=640, help="Input shapes")
    # Set TVM tuning mode, default is "autotvm"
    parser.add_argument("--tuning-mode", type=str, default="autotvm", help="Tuning mode")
    # Set device key
    parser.add_argument("--device-key", type=str, help="Device key")
    # Change number of threads that can be used for tuning
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads")
    # Compute time it takes to tune each task
    parser.add_argument("--time-tasks", action="store_true", help="Compute time it takes to tune each task")
    # Set tuning operators to be used in tasks
    parser.add_argument("--ops", type=str, default=None, help="Tuning operators")
    
    args = parser.parse_args()
    
    if args.version:
        check_tvm_version()
        exit(0)
        
    if args.download_model:
        # Download a Resnet50 Model from ONNX Model Zoo
        
        # Download the model
        model_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"
        subprocess.run(["wget", "-O", "models/resnet50-v2-7.onnx", model_url])
        
    # Check model_name is set
    if not args.model_name:
        print("Please set model name with --model-name")
        exit(1)
    # Check model_path is set
    if not args.model_path:
        print("Please set model path with --model-path")
        exit(1)
    # Check device key is set
    if not args.device_key:
        print("Please set device key with --device-key")
        exit(1)
        
    # Enter program - perform basic checks
    
    if args.ops:
        ops = args.ops.split(",")
        ops = (relay.op.get(op) for op in ops)
        

    model_path = MODELS / args.model_path
    print("Using model from: ", model_path)
    
    model_name = args.model_name
    print("Using model name: ", model_name)
    
    device_key = args.device_key
    print("Using device key: ", device_key)
    
    num_threads = args.num_threads
    os.environ["TVM_NUM_THREADS"] = str(num_threads)
    cmd = "echo $TVM_NUM_THREADS"
    print("TVM_NUM_THREADS: ", subprocess.check_output(cmd, shell=True).decode("utf-8"))
    
    target = args.target
    print("Target device: ", target)
        
    
    # Load ONNX model
    print("Loading model: ", model_name)
    onnx_model = onnx.load(model_path)
    
    # Model metadata - CV only
    batch_size = args.batch_size
    H = args.img_size
    W = args.img_size
    C = 3
    input_shape = (batch_size, C, H, W)
    input_dtype = "float32"
    input_name = "input"
    inputs={onnx_model.graph.input[0].name: input_shape}
    print("The input shape is: ", input_shape)
    
    # Create logging file
    print("Tuning mode: ", args.tuning_mode)
    logfile_name = create_logging_file(model_name, tuning_method=args.tuning_mode)
    print(logfile_name)
    
    # Get module and params from the model
    print("Extracting module and params from ONNX model...")
    tasks = extract_graph_tasks(onnx_model, target=args.target, target_host=args.target_host,ops=ops)
    print("Number of tasks: ", len(tasks))
    print("Tasks: ", tasks)
    
    # Runing tuning tasks
    print("Tuning...")
    tune_model_tasks(tasks, logfile_name, device_key, num_threads=None, num_measure_trials=args.trials)
    
        