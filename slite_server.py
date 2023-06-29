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
import time
from deepcopy import deepcopy

def process_pending_job(job, resources):
    """
    This function takes the pending jobs and the free gpus and assigns the jobs to the gpus.
    """

    # get the free gpus
    free_gpu_ids = deepcopy(resources.summary.free_gpu_ids)

    # get the user name
    user = job.config["user"]
    
    # find a node that has enough free gpus
    allocated_node = None
    allocated_gpu_ids = None
    for node in free_gpu_ids:
        if len(free_gpu_ids[node]) >= job.config["gres"]:
            allocated_node = node
            allocated_gpu_ids = free_gpu_ids[node][:job.config["gpus"]]
            break

    # if no node has enough free gpus, then continue
    if allocated_node is None:
        return


    # remove the allocated gpus from the free gpus
    free_gpu_ids[allocated_node] = free_gpu_ids[allocated_node][job.config["gpus"]:]
    # update the free gpus
    resources.summary.free_gpu_ids = free_gpu_ids
    # update the resources
    resources.update()

    # update the job with the allocated node
    job.config["allocated_node"] = allocated_node
    # update the job with the gpu id
    job.config["allocated_gpu_ids"] = allocated_gpu_ids
    # update the job with the start time
    job.config["start_time"] = time.time()
    # update the job state
    job.state = "configuring"
    # update the job with the new state
    job.update()


def process_pending_jobs(job_list, resources):
    """
    This function ranks the pending jobs based on:
        (1) gpu requirement
        (2) user gpu usage.
    Then it processes the jobs in order.
    """

    gpu_per_user = deepcopy(resources.summary.gpu_per_user)

    job_list.sort(key=lambda x: gpu_per_user[x.config["user"]], reverse=True)
    job_list.sort(key=lambda x: x.config["gres"], reverse=True)

    for job in job_list:
        process_pending_job(job, resources)


def process_running_job(job, resources):
    """
    This function takes the running jobs and the gpu log and checks if the job is still running.
    If the job is not running anymore, then it frees up the gpus.

    If the job is still running, but the time limit is exceeded, then it kills the job.
    """

    # get the free gpus
    free_gpu_ids = deepcopy(resources.summary.free_gpu_ids)

    # get the allocated node
    allocated_node = job.config["allocated_node"]
    # get the allocated gpus
    allocated_gpu_ids = job.config["allocated_gpu_ids"]

    # check if the job is still running
    if job.state == "running":
        # check the running time of the job
        if job.config["time_limit"] < time.time() - job.config["start_time"]:
            job.state = "terminated"
            job.update()
            # the job is terminated, but we wait for the client to kill the job
            # as only the client process has the privileges to kill the job
            return

    if job.state == "finished":
        # if the job is not running anymore, then free up the gpus
        free_gpu_ids[allocated_node] = free_gpu_ids[allocated_node] + allocated_gpu_ids
        # update the free gpus
        resources.summary.free_gpu_ids = free_gpu_ids
        # update the resources
        resources.update()

        # update the job state
        job.state = "finished"
        # update the job with the new state
        job.update()

def process_running_jobs(job_list, resources):
    """
    This function checks the running jobs and:
        (1) sends out the kill signal to the client if the time limit is exceeded
        (2) frees up the gpu log if the job is finished
    """
    for job in job_list:
        process_running_job(job, resources)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLURM Lite Server')
    parser.add_argument("--resource-run", type=str, default="csbotos/torrnode-monitor/0hmqk54d", help="wandb run that tracks the resources")
    parser.add_argument("--jobs-project", type=str, default="csbotos/torrnode-jobs", help="wandb project that tracks the submitted jobs")
    args = parser.parse_args()


    # scan all the runs in the 
    jobs = wandb.Api().runs(args.jobs_project)
    resources = wandb.Api().runs(args.resource_run)

    gpu_per_user = {}
    free_gpu_ids = deepcopy(resources.summary["free_gpu_ids"])


    # select the runs that are pending
    pending_jobs = [job for job in jobs if job.state == "pending"]

    # select the runs that are running
    running_jobs = [job for job in jobs if job.state == "running"]

    # process the pending jobs
    process_pending_jobs(pending_jobs, resources)

    # process the running jobs
    process_running_jobs(running_jobs, resources)

    





    