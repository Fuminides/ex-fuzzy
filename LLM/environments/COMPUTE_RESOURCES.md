There are three places where the scripts can be running: a Surface laptop, a workstation and the Essex HPC environment. 

Depending on the scope of the run, some experiments and scales are more suitable. You should not try to run very expensive experiments in the laptop and you should not defer trivial things to the workstation or the hpc.

## Specs of each:

### CERES (hpc) 

The Ceres cluster is a computational cluster built using the Rocks Clustering Solution with CentOS Linux. It has been built using a mix of Fujitsu (servers), Dell (servers), Mellanox (switches) and HP (switches) hardware.

For computational purposes, the cluster has 1632 64 bit processing cores (3232 with hyperthreading) using a mix of Intel E5-2698, Intel Gold 5115, 6152 & 6238L CPUs on a number of dedicated servers each with between 500Gb & 6Tb RAM. Storage is provided by a set of storage nodes (servers with disk enclosures) providing 1542Tb of storage. Inter-node connectivity is via 10GbE switches which provide a private network exlusively used by the cluster. There are also 24 NVidia GTX & RTX Series GPU cards (16 x GTX1080Ti & 8 x RTX2080) attached via dedicated GPU servers for research purposes.

Job scheduling is handled by the Open Grid Engine scheduler and users have the option of batch processing or interactive node use.

### SUSHI (laptop)
(Surface laptop 6 for businesses)

Windows 11 Pro 64-bit
CPU: Intel(R) Core(TM) Ultra 7, ~3.8GHz
RAM: 16GB RAM

GPU: Not Nvidia (so no cuda)
It is a laptop integrated Intel graphics card. 
Intel(R) Arc(TM) Graphics 9GB Memory
    
### BURRITA (desktop)
(A Ubuntu workstation of moderate power)

Ubuntu OS.
CPU: AMD Ryzen 5 5600X 6-core Processor
RAM: 16GB
GPU: Geforce RTX 5060 Ti 8GB Memory



