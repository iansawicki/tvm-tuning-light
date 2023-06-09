import tvm 
import os
import argparse
import subprocess
from pathlib import Path
from tvm import relay, autotvm, auto_scheduler
import onnx
from datetime import datetime as dt
import platform
import time

#Set parent base directory
BASE_PATH = Path(__file__).parent

# Create /models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")
    
# Create /tuning_logs directory if it doesn't exist
if not os.path.exists("tuning_logs"):
    os.makedirs("tuning_logs")
    
    
# Default path to models
MODELS = BASE_PATH / "models"
TUNING_LOGS = BASE_PATH / "tuning_logs"


# Check TVM version 
def check_tvm_version():
    print(tvm.__version__)
    
# Default TVM tuning options
num_measure_trials = 1000
opt_level = 3
verbose = True

# Default RPC options
# number: is the number of times to run the generated code for taking an average
# and repeat: is the number of times to repeat the measurement. 
# The generated code will run (1 + number x repeat) times, where the first “1” is warm up and will be discarded.

# enable_cpu_cache_flush: is a flag to enable cache flush before each measurement.
rpc_runner = dict(host="127.0.0.1",
              port=9190,
              timeout=1000,
              number=1,
              repeat=None,
              min_repeat_ms=300,
              enable_cpu_cache_flush=False)

# What's the difference between number and repeat?

# Default tuning options
early_stopping = 300 # When to stop tuning when not finding better results
tuner =  tvm.autotvm.tuner.XGBTuner
tuner_settings = dict(loss_type="rank", feature_type="knob")

  
def create_logging_file(model_name, tuning_method="autotvm"):
    
    # Build tuning log path name
    epoch_time = str(dt.now().timestamp()).split(".")[0]
    platform_arch = platform.machine()
    vendor_name = platform.node()
    logfile = TUNING_LOGS / f'{tuning_method}_{model_name}_{platform_arch}_{vendor_name}_{epoch_time}.log'
    return str(logfile)

def extract_graph_tasks(onnx_model, target="llvm", *args, **kwargs):
    # Extract tasks from the graph
    mod, params = relay.frontend.from_onnx(onnx_model, shape=inputs, dtype={input_name: input_dtype},opset=11)
    print("Extract tasks...")
    if 'ops' in kwargs.keys() and kwargs['ops'] != '':
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params, ops=kwargs["ops"])
        print("Extracting tasks for the following ops: ", kwargs["ops"])
    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
        
    return tasks

def extract_graph_tasks_autoscheduler(onnx_model, target="llvm", *args, **kwargs):
    # Extract tasks from the graph
    mod, params = relay.frontend.from_onnx(onnx_model, shape=inputs, dtype={input_name: input_dtype},opset=11)
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target=target)
    return tasks, task_weights

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end-start} seconds to execute.")
        return result
    return wrapper
    

@time_it
def tune_model_tasks(tasks, logfile_name, device_key , *args, **kwargs):
     # create tmp log file
    logfile_name = str(logfile_name)
    tmp_log_file = logfile_name + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    print(tmp_log_file)
    
    # Create builder, runner, and measure_option
    builder = autotvm.LocalBuilder(timeout=10)
    runner=autotvm.RPCRunner(device_key,**rpc_runner)
    measure_option = autotvm.measure_option(builder=builder, runner=runner)
    
    # Tune tasks
    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        
        # Create tuner_obj
        tuner_obj = tuner(task, **tuner_settings)

        # Process_tuning
        tsk_trial = min(num_measure_trials, len(task.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            verbose=verbose,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, logfile_name)
    os.remove(tmp_log_file)
    
def tune_model_tasks_autoscheduler(tasks, task_weights, logfile_name, device_key , *args, **kwargs):
    # create tmp log file
    logfile_name = str(logfile_name)
    tmp_log_file = logfile_name + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    print(tmp_log_file)
    
    # Create runner for auto_scheduler
    runner = auto_scheduler.RPCRunner(device_key,**rpc_runner)
    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trials,
        runner=runner,
        early_stopping=early_stopping,
        measure_callbacks=[auto_scheduler.RecordToFile(tmp_log_file)],
        verbose=2
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_options)
    
    # pick best records to a cache file
    auto_scheduler.ApplyHistoryBest(tmp_log_file).to_file(logfile_name)
    os.remove(tmp_log_file)
   
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Check version
    parser.add_argument("--version", action="store_true", help="Check TVM version")
    # Download toy model
    parser.add_argument("--download-model", action="store_true", help="Download a toy model")
    # Speficy model to download
    parser.add_argument("--download-model-name", type=str, default="resnet50", help="Model to download")
    # Set path to model
    parser.add_argument("--model-path", type=str, help="Path to model")
    # Set model name
    parser.add_argument("--model-name", type=str, help="Name of model")
    # Set batch_size, default is 1
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    # Override input shapes
    parser.add_argument("--insert-shapes", type=str, help="Override input shapes")
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
    parser.add_argument("--ops", type=str, default="", help="Tuning operators")
    
    args = parser.parse_args()
    
    if args.version:
        check_tvm_version()
        exit(0)
        
    if args.download_model:
        RESNET = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx"
        MNIST = "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-1.onnx"        
        # Download the model
        if args.download_model_name == "resnet50":
            model_url = RESNET
            download_model_name = MODELS / "resnet50-v2-7.onnx"
        elif args.download_model_name == "mnist":
            model_url = MNIST
            download_model_name = MODELS / "mnist-1.onnx"
        subprocess.run(["wget", "-O", download_model_name, model_url])
        
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
    
    ops = args.ops.split(",")
    if len(ops) > 1:
        ops = tuple([relay.op.get(op) for op in ops])
        
    else:
        ops = ""
                
    model_path = args.model_path
    print("Using model from: ", model_path)
    
    model_name = args.model_name
    print("Using model name: ", model_name)
    
    device_key = args.device_key
    print("Using device key: ", device_key)
    
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
    if args.insert_shapes:
        input_shape = args.insert_shapes.split(",")
        input_shape = tuple([int(i) for i in input_shape])

    input_dtype = "float32"
    input_name = "input"
    inputs={onnx_model.graph.input[0].name: input_shape}
    print("The input shape is: ", input_shape)
    
    # Create logging file
    logfile_name = create_logging_file(model_name, tuning_method=args.tuning_mode)
    
    if args.tuning_mode=="autotvm":
    
      
        print("Tuning mode: ", args.tuning_mode)
        print(logfile_name)
        
        # Get module and params from the model
        print("Extracting module and params from ONNX model...")
        tasks = extract_graph_tasks(onnx_model, ops=ops, target=args.target, target_host=args.target_host)
        print("Number of tasks: ", len(tasks))
        #print("Tasks: ", tasks)
        
        # Runing tuning tasks
        print("Tuning...")
        tune_model_tasks(tasks, logfile_name, device_key, num_measure_trials=args.trials)
        
    elif args.tuning_mode=="autoscheduler":
        print("Tuning mode: ", args.tuning_mode)
        print(logfile_name)
        
        # Get module and params from the model
        print("Extracting module and params from ONNX model...")
        tasks, task_weights = extract_graph_tasks_autoscheduler(onnx_model, ops=ops, target=args.target, target_host=args.target_host)
        print("Number of tasks: ", len(tasks))
        #print("Tasks: ", tasks)
        
        # Run tuning tasks
        print("Tuning...")
        tune_model_tasks_autoscheduler(tasks, task_weights, logfile_name, device_key, num_measure_trials=args.trials)
        
        
    
    
        