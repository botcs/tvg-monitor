# Resource monitoring and job scheduling for TorrNodes

This repository contains:
- resource **monitoring** tools
  - compute node availability
  - free GPU IDs per nodes
  - allocated GPU IDs per users
  - Shared storage
    - I/O speed
    - Free Space
  - Local storage
    - I/O speed
    - Free Space  
- **resource allocation** and **job scheduling** server
- **job managing** client

---

The resource monitoring process runs independently from the scheduler server and client.
The client takes the arguments:
- user
- script to run
- resources
- time limit for the job

The jobs are submitted to an online database (via [WandB](https://wandb.ai/)) from which the server process takes the jobs and:
- performs ranking based on user statistics
- scans the available resources
- allocates resources to pending jobs
- terminates overdue running jobs
- frees up resources of finished jobs
