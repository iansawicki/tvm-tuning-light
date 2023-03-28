# File to manage creation of RPC server and device registration
import os, argparse


def start_rpc_server():
    cmd = f"python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190"
    os.system(cmd)
      
def register_device(key):
    cmd = f"python -m tvm.exec.rpc_server --tracker 0.0.0.0:9190 --port 9090 --key {key} --no-fork"
    os.system(cmd)

def query_devices():
    cmd = "python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190"
    os.system(cmd)
    


def show_rpc_server():
    cmd = "ps -ef | grep rpc_tracker"
    os.system(cmd)
    
def parse_rpc_pid():
    cmd = "ps -ef | grep rpc_tracker | grep -v grep | awk '{print $2}'"
    pid = os.popen(cmd).read()
    return(pid)

def kill_rpc_server(port=9190):
    cmd = f"kill $(lsof -t -i:{port})"
    os.system(cmd)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true", help="Start RPC server")
    parser.add_argument("--register-device-key", type=str, help="Register device")
    parser.add_argument("--query", action="store_true", help="Query devices")
    parser.add_argument("--kill-rpc", action="store_true", help="Kill RPC server")
    parser.add_argument("--show-rpc", action="store_true", help="Show RPC server")
    args = parser.parse_args()
    
    if args.start:
        start_rpc_server()
    elif args.register_device_key:
        register_device(args.register_device_key)
    elif args.query:
        query_devices()
    elif args.kill_rpc:
        kill_rpc_server()
    elif args.show_rpc:
        parse_rpc_pid()
    else:
        print("Invalid option")