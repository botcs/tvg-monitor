"""
SLURM Lite Client
for TVG internal use only
written by Csabi

This is a client for the SLURM Lite Server.
It is used to submit jobs to the server.

It delegates the logging to the wandb library to keep track of the jobs.

Each job is represented with a wandb run.
"""

import wandb
import subprocess
import sys
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", type=str, help="Path to the script to run.")
    parser.add_argument("--user", type=str, required=True, help="User name.")
    parser.add_argument("--gres", default=1, type=int, help="Number of GPUs to request.")
    parser.add_argument("--time-limit", default=24, type=int, help="Number of hours to request.")
    args = parser.parse_args()
    
    wandb.init(
        project="tvg-slurm-lite", 
        job_type="slurm-lite-client",
        entity="tvg",
        config={
            "script_path": args.script_path,
            "state": "pending",
            "user": args.user,
            "gres": args.gres,
            "time_limit": args.time_limit,
            "allocated_node": None,
            "allocated_gpus": None,
        }
    )
    print(f"Queued job: {wandb.run.id}")

    # Wait until the server assigns a node to this job.
    while wandb.config.allocated_node is None:
        time.sleep(1)
        wandb.config.update()

    # Run the script on the allocated node.
    cmd = [
        "ssh", wandb.config.allocated_node, 
        f"CUDA_VISIBLE_DEVICES={wandb.config.allocated_gpus}",
        "bash", wandb.config.script_path
    ]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        # set the job state to running
        wandb.config.state = "running"
        wandb.config.update()
        print(f"Running process: {proc.pid} on node: {wandb.config.allocated_node}")

        while True:
            try:
                # check if the process is set to be terminated
                if wandb.config.state == "terminated":
                    print(f"Terminating process: {proc.pid} on node: {wandb.config.allocated_node}")
                    proc.terminate()
                    time.sleep(30)
                    if proc.poll() is None:
                        proc.kill()
                    break

                output = proc.communicate(timeout=1)
                print(output)
                if proc.poll() is not None:
                    break
                time.sleep(1)
            except subprocess.TimeoutExpired:
                pass

        # set the job state to finished
        wandb.config.state = "finished"
        wandb.config.update()
        print(f"Finished job: {wandb.run.id}")


        


        

        












