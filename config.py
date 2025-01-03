import torch

# define configuration modes
MODE_DPSGD_B = 1
MODE_DPSGD_R = 2
MODE_DPSGD_F = 3
MODE_SGD = 4
MODE_EANA = 5

# set training configurations
disable_poisson_sampling = False
num_gathers = 20
num_gathers_list = None
dpsgd_mode = MODE_DPSGD_B
batch_size = 4
cur_batch_size = 4
data_size = 1

# set device
use_cpu = True # True for CPU-GPU systems, False for GPU-only systems
device = torch.device('cpu')

# Profiler
profiler = None

# PyTorch custom functions
coalesce_nthreads = 32
coalesce_optimize = "baseline" # "baseline" / "multi_thread_openmp" / "multi_thread_embeddingbag"

noise_base_nthreads = 32
noise_base_optimize = "multi_thread" # "baseline" / "multi_thread"
noise_final_nthreads = 32 # always execute in multi-threaded manner

# when this variable sets to 1,
# 1) add 1 instead of noise
# 2) do all delayed noise updates after final training iteration
is_debugging = False
debugging_type = "without_noise"

unique_optimize = "baseline" # "baseline" / "multi_thread"
