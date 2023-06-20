""" 
monitor the availability of torrnode[1-15]
for TVG internal use only
written by Csabi

Watch the following metrics:
- number of GPUs being used by each user
- free space on mountpoints
- IO speed on mountpoints

The mountpoints are (as of 2023-06-20, from torrnode8):
- Shared storage accessible from all nodes
    - /homes/53
    - /oldhomes/53
    - /storage
    - /storage2
    - /scratch/network/ssd
    - /scratch/network/ssd2
    - /scratch/network/ssd3
    - /scratch/network/ssd4

- Local storage accessible only from the compute node
    - /scratch/local/ssd

Note that:

We are using torrnode8 as the representative node
torrnode8 is the first node of torrstore2, hosting torrnode8-15
reported IO speed might be different from torrstore1, hosting torrnode1-7

each of the mountpoints has a folder for each user, e.g. /storage/<username>
and in this script reports with "csbotos" as the username

for some reason /homes and /oldhomes is segmented into ./53 and ./55
but luckily they are mounted on the same drive, so we will stick to ./53
"""

import wandb 
import subprocess 
import argparse 
from collections import Counter 
import time
import json
import logging

parser = argparse.ArgumentParser() 
parser.add_argument("--resume-run-id", type=str) 
parser.add_argument("--io-time-delta", type=int, default=600)
parser.add_argument("--df-time-delta", type=int, default=600)
parser.add_argument("--gpu-time-delta", type=int, default=600)
parser.add_argument("--state-time-delta", type=int, default=60)
parser.add_argument("--timeout", type=int, default=60)
args = parser.parse_args()

torrnodes = [
    "torrnode1",
    "torrnode2",
    "torrnode3",
    "torrnode4",
    "torrnode5",
    "torrnode6",
    "torrnode7",
    "torrnode8",
    "torrnode9",
    # "torrnode10", # used by IT for experiments
    "torrnode11",
    "torrnode12",
    "torrnode13",
    "torrnode14",
    "torrnode15",
]



SHARED_MOUNTPOINTS = [
    "/homes/53",
    "/oldhomes/53",
    "/storage",
    "/storage2",
    "/scratch/network/ssd",
    "/scratch/network/ssd2",
    "/scratch/network/ssd3",
    "/scratch/network/ssd4",
]

LOCAL_MOUNTPOINTS = [
    "/scratch/local/ssd",
]

# add user to the mountpoints
reference_user = "csbotos"
SHARED_MOUNTPOINTS = [mountpoint + "/" + reference_user for mountpoint in SHARED_MOUNTPOINTS]
LOCAL_MOUNTPOINTS = [mountpoint + "/" + reference_user for mountpoint in LOCAL_MOUNTPOINTS]

def gather_gpu_info(tn_users):
    """
    Using torrnode[1-15] check the number of GPUs being used by each user

    Return a Counter mapping username to number of GPUs being used
    """
    usage = Counter({user: 0 for user in tn_users})
    free_gpus = Counter({node: 0 for node in torrnodes})
    for node in torrnodes:
        node_command = f"ssh {node} gpustat --json"
        logging.debug(f"Running command: {node_command}")
        try:
            node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8')
            logging.debug(f"Output: {node_output}")
        except subprocess.CalledProcessError:
            logging.warning(f"Couldn't connect to {node}")
            continue
        stats = json.loads(node_output)
        for gpu in stats["gpus"]:
            if len(gpu["processes"]) == 0:
                free_gpus[node] += 1
            for proc in gpu["processes"]:
                username = proc["username"]
                usage[username] += 1

    logging.info("Result of query:")
    logging.info(f"GPU usage: {usage}")
    logging.info(f"Free GPUs: {free_gpus}")
    return usage, free_gpus


def gather_df_info(convert_to_gb=True):
    """
    Using torrnode8 as a default representative node, check the free space on each mountpoint

    Return a Counter mapping mountpoint to number of nodes with available free space
    """

    def run_ssd_command(node, mountpoints):
        free_space = {mountpoint: 0 for mountpoint in mountpoints}

        if node not in torrnodes:
            logging.warning(f"{node} is not a valid node")
            return free_space
        
        node_command = f"ssh {node} df {' '.join(mountpoints)} --output=avail | tail -n +2"
        logging.debug(f"Running command: {node_command}")
        try:
            node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8').strip()
            logging.debug(f"Output: {node_output}")
            avail = node_output.split("\n")
            if len(avail) != len(mountpoints):
                logging.warning(f"{node} returned {len(avail)} values, expected {len(mountpoints)}")
                return free_space
            avail = [int(x) for x in avail]
        except subprocess.CalledProcessError:
            logging.warning(f"Couldn't connect to {node}")
            return free_space
        except ValueError:
            logging.warning(f"Couldn't parse output from {node}")
            return free_space

        for i, mountpoint in enumerate(mountpoints):
            free_space[mountpoint] = avail[i]
        return free_space
    
    # start with torrnode8, and exit as soon as we find a node that's responding
    for i in range(8, 16):
        free_space = Counter()
        free_space.update(run_ssd_command(f"torrnode{i}", SHARED_MOUNTPOINTS))
        if len(free_space) != len(SHARED_MOUNTPOINTS):
            logging.warning(f"torrnode{i} returned {len(free_space)} values, expected {len(SHARED_MOUNTPOINTS)}")
            continue
        else:
            break
    
    # if we get here, then we have a working node
    # now we can query the rest of the nodes for local storage
    for node in torrnodes:
        # update keys with the node id
        local_free_space = run_ssd_command(node, LOCAL_MOUNTPOINTS)
        for mountpoint in local_free_space:
            free_space[f"{node}:{mountpoint}"] = local_free_space[mountpoint]
    
    # remove the reference user from the mountpoints
    free_space = {
        mountpoint.replace(f"/{reference_user}", ""): free_space[mountpoint] 
        for mountpoint in free_space
    }

    if convert_to_gb:
        # assuming the output is in KB, convert to GB but with 2^20 instead of 10^9
        # to match the output of df if -h was used... don't ask me why
        free_space = {mountpoint: free_space[mountpoint] / 2**20 for mountpoint in free_space}

    logging.info("Result of query:")
    logging.info(free_space)
    return free_space


def gather_io_info():
    """
    Using torrnode[1-15] check the number of IO speed of each mountpoint

    Return a Counter mapping the mountpoint to the average IO speed
    """
    def run_ssd_command(node, mountpoints, num_files=1e3):
        io_speed = {}

        if node not in torrnodes:
            logging.warning(f"{node} is not a valid node")
            return io_speed
        
        # since iostat doesn't function as expected, we'll use dd to get the speed

        for mountpoint in mountpoints:
            logging.debug(f"Checking IO times on {node}:{mountpoint}")
            num_files = int(num_files)
            write_command = f"ssh {node} dd if=/dev/zero of={mountpoint}/.test bs=1M count={num_files} 2>&1 | tail -n 1"
            read_command = f"ssh {node} dd if={mountpoint}/.test of=/dev/null bs=1M count={num_files} 2>&1 | tail -n 1"
            try:
                logging.debug(f"Running command: {read_command}")
                read_output = subprocess.check_output(read_command, shell=True, timeout=args.timeout).decode('utf-8').strip()
                logging.debug(f"Output: {read_output}")

                logging.debug(f"Running command: {write_command}")
                write_output = subprocess.check_output(write_command, shell=True, timeout=args.timeout).decode('utf-8').strip()
                logging.debug(f"Output: {write_output}")

                # parse the output
                # 1048576000 bytes (1.0 GB, 1000 MiB) copied, 0.889669 s, 1.2 GB/s
                #                     We need to get this number ^
                read_time = float(read_output.split(" ")[-4])
                write_time = float(write_output.split(" ")[-4])
                logging.debug(f"Read time: {read_time}, write time: {write_time}")
                
                # convert to GB/s...
                # again with the conversion dd is not consistent, 
                # they report both KiB and KB and GB but using 1GB 
                # using 1/time to get the speed doesn't result in the same number
                # as the one reported by dd (but it's close enough)
                read_speed = 1 / read_time
                write_speed = 1 / write_time

            except subprocess.CalledProcessError:
                logging.warning(f"Couldn't connect to {node}")
                return io_speed
            except ValueError:
                logging.warning(f"Couldn't parse output from {node}")
                return io_speed
            
            io_speed[f"{node}:{mountpoint}"] = (read_speed, write_speed)

        return io_speed
                    

    # start with torrnode8, and exit as soon as we find a node that's responding
    for i in range(8, 16):
        io_speeds = {}
        io_speeds.update(run_ssd_command(f"torrnode{i}", SHARED_MOUNTPOINTS))
        if len(io_speeds) != len(SHARED_MOUNTPOINTS):
            logging.warning(f"torrnode{i} returned {len(io_speeds)} values, expected {len(SHARED_MOUNTPOINTS)}")
            continue
        else:
            break
    
    # if we get here, then we have a working node
    # now we can query the rest of the nodes for local storage
    for node in torrnodes:
        io_speeds.update(run_ssd_command(node, LOCAL_MOUNTPOINTS))


    # remove the reference user from the mountpoints
    io_speeds = {
        mountpoint.replace(f"/{reference_user}", ""): io_speeds[mountpoint]
        for mountpoint in io_speeds
    }

    # split the read and write speeds
    read_speeds = {mountpoint: io_speeds[mountpoint][0] for mountpoint in io_speeds}
    write_speeds = {mountpoint: io_speeds[mountpoint][1] for mountpoint in io_speeds}

    logging.info("Result of query:")
    logging.info(f"Read speeds: {read_speeds}")
    logging.info(f"Write speeds: {write_speeds}")
    
    return read_speeds, write_speeds

def check_state():
    """
    Check the state of all the nodes and return a dictionary mapping the node name to a boolean
    indicating if the node is online or not
    """
    status = {}
    for node in torrnodes:
        try:
            node_command = f"ssh {node} echo 'online'"
            logging.debug(f"Running command: {node_command}")
            node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8').strip()
            logging.debug(f"Node {node} responded with {node_output}")
            if node_output.strip() == "online":
                status[node] = 1
            else:
                status[node] = 0
        except subprocess.CalledProcessError:
            status[node] = 0
    
    logging.info("Node status:")
    logging.info(status)
    return status


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s-[%(levelname)s]: %(message)s', level=logging.DEBUG)

    # list all the users in /homes/53 and /homes/55 through torrnode8
    node_command = f"ssh torrnode8 ls /homes/53"
    node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8')
    tn_users = node_output.split("\n")[:-1]
    node_command = f"ssh torrnode8 ls /homes/55"
    node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8')
    tn_users += node_output.split("\n")[:-1]
    node_command = f"ssh torrnode8 ls /oldhomes/53"
    node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8')
    tn_users += node_output.split("\n")[:-1]
    node_command = f"ssh torrnode8 ls /oldhomes/55"
    node_output = subprocess.check_output(node_command, shell=True, timeout=args.timeout).decode('utf-8')
    tn_users += node_output.split("\n")[:-1]

    
    # resume wandb session if it's already existing
    if args.resume_run_id is not None:
        print("Resuming run", args.resume_run_id)
        wandb.init(
            id=args.resume_run_id,
            project="torrnode-monitor",
            entity="csbotos",
            resume=True,
        )
   
    else:
        wandb.init(
            project="torrnode-monitor", 
            entity="csbotos",
        )
    
    wandb.config.update({tn_user: True for tn_user in tn_users})

    # track how much time passed since checking
    last_io_time = 0
    last_df_time = 0
    last_gpu_time = 0
    last_state_time = 0

    IO_TIME_DELTA = args.io_time_delta
    DF_TIME_DELTA = args.df_time_delta
    GPU_TIME_DELTA = args.gpu_time_delta
    STATE_TIME_DELTA = args.state_time_delta

    while True:
        try: 
            # check the IO speed
            if time.time() - last_io_time > IO_TIME_DELTA:
                read_speeds, write_speeds = gather_io_info()
                wandb.log({"read_speeds": read_speeds, "write_speeds": write_speeds})
                wandb.summary["read_speeds"] = read_speeds
                wandb.summary["write_speeds"] = write_speeds
                last_io_time = time.time()

            # check the disk usage
            if time.time() - last_df_time > DF_TIME_DELTA:
                free_space = gather_df_info()
                wandb.log({"free_space": free_space})
                wandb.summary["free_space"] = free_space
                last_df_time = time.time()

            # check the GPU usage
            if time.time() - last_gpu_time > GPU_TIME_DELTA:
                gpu_usage, free_gpus = gather_gpu_info(tn_users)
                wandb.log({"gpu_usage": gpu_usage})
                wandb.log({"free_gpus": free_gpus})
                wandb.summary["gpu_usage"] = gpu_usage
                wandb.summary["free_gpus"] = free_gpus
                last_gpu_time = time.time()

            # check the state of the nodes
            if time.time() - last_state_time > STATE_TIME_DELTA:
                node_state = check_state()
                wandb.log({"node_state": node_state})
                wandb.summary["node_state"] = node_state
                last_state_time = time.time()
        finally:
            # sleep for a bit
            time.sleep(10)
