"""
SLURM Lite Server
for TVG internal use only
written by Csabi

This is a server for the SLURM Lite Client.
It takes jobs from the client and assigns them to the free gpus.

It delegates the logging to the wandb library to keep track of the jobs.
Each job is represented with a wandb run.

The server uses the wandb API to track the queued jobs and the free resources.
"""

import wandb
import argparse
from deepcopy import deepcopy

def find_resource_for_job(job, free_gpus):
    """
    Find a resource for the job.
    """
    
def allocate_resource_for_job(job, free_gpus):
    """
    Allocate a resource for the job.
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLURM Lite Server')
    parser.add_argument("--resource-run", type=str, default="csbotos/torrnode-monitor/0hmqk54d", help="wandb run that tracks the resources")
    parser.add_argument("--jobs-project", type=str, default="csbotos/torrnode-jobs", help="wandb project that tracks the submitted jobs")
    args = parser.parse_args()


    # scan all the runs in the 
    jobs = wandb.Api().runs(args.jobs_project)
    resources = wandb.Api().runs(args.resource_run)

    gpu_per_user = {}
    free_gpus = deepcopy(resources.summary["free_gpus"])


    # select the runs that are pending
    pending_jobs = [job for job in jobs if job.state == "pending"]

    # select the runs that are running
    running_jobs = [job for job in jobs if job.state == "running"]

    





    