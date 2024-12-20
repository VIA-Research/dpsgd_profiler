# Profiler for Differentially Private Training of ML models
This GitHub repository provides efficient and accurate performance profiling tools designed for differentially private training of machine learning (ML) models. Specifically, we focus on Differentially Private Stochastic Gradient Descent (DP-SGD), an algorithm that extends standard SGD training.

Since this repository is built on the Opacus library (which, as of version 1.2.0, only supports naive DP-SGD, referred to as DP-SGD(B)), we extend it to support reweighted DP-SGD and fast DP-SGD, referred to as DP-SGD(R) and DP-SGD(F), respectively.

Additionally, the profiler included in this repository is specifically tailored for DLRM training, one of the most widely used ML models in both industry and academia. Users who wish to profile other types of models can also utilize this tool by making adjustments to the profiler code.

## Setup
Since testing this profiler generates DLRM models that can reach hundreds of gigabytes in size, first select a directory where a large amount of model data can be stored. (e.g., `/raid/model_weight/`).
```bash
mkdir {absolute_path_to_model_weight_directory}
export PATH_MODEL_WEIGHT={absolute_path_to_model_weight_directory}
```

Then, create docker container from official docker image.
```bash
docker run -ti --gpus all --shm-size 5g --name lazydp -v $PATH_MODEL_WEIGHT:/model_weight -e "PATH_MAIN=/workspace/dpsgd_profiler" -e "PATH_MODEL_WEIGHT=/model_weight" --cap-add SYS_NICE nvcr.io/nvidia/pytorch:23.09-py3
```

In the docker container, clone this repository.
```bash
cd /workspace
git clone https://github.com/VIA-Research/dpsgd_profiler.git
```

Run `setup.sh` in cloned directory to extend PyTorch and install required packages. This extension accelerates several operations such as noise generation and sparse parameter updates on CPU device.
```bash
cd $PATH_MAIN
./setup.sh
```

## How to Profile Differentially Private Training with Our Tool
In the `$PATH_MAIN/custom_utils.py` file, a Python class named `LatencyMeter` is implemented. This tool predefines the names of operations whose latencies are to be measured, supporting up to two levels (as explained in the comments of this Python file).

Here is an example assuming the user profiles DLRM training with DP-SGD(B).
```python
# Latency
profiler = LatencyMeter("dpsgd_b", "path_for_results", ${number_of_iterations}, "name_of_result_files")
...
# code section 1 start
profiler.start("code_section_1")
for i in range(n):
    profiler.start_l2("code_section_1_1")
    # code section 1-1
    profiler.end_l2("code_section_1_1")

    profiler.start_l2("code_section_1_2")
    # code section 1-2
    profiler.end_l2("code_section_1_2")

# code section 1 end
profiler.end("code_section_1")
```

In the `$PATH_MAIN/bench` directory, there are two shell scripts, `run_with_emb_scaling.sh` and `run_test.sh`, to use this profiler for DLRM models with various configurations. 

Simply, run the codes as bellow:
```bash
cd $PATH_MAIN/bench
./run_with_emb_scaling.sh

# or
cd $PATH_MAIN/bench
./run_test.sh
```

## Profiling result
The result csv/txt files will be generated in `$PATH_MAIN/result`.

- (MAIN RESULT) In `$PATH_MAIN/result/merged_result` directory, generated csv file is the main result.
    - Each column represents the result for a certain model/training configuration.
    - First four rows represent the latency breakdown for 3 parts of model training.
        1. Forward propagation
        2. First backpropagation
        3. Second backpropagation
        4. Model update
    - Next six rows represent the detailed latency breakdown of model update stage.
    - Last row is to measure the time for executing certain part of code you want to examine. (If not required, just ignore)
- In `$PATH_MAIN/result/log` directory, text files (e.g., `{AAA}_{BBB}_s_{CCC}_B_{DDD}_L_{EEE}_{FFF}.txt`) are generated for each configuration and these files are the log file recording the execution time and shows the training progress.
    - `{AAA}`: model type
        1. basic: model whose tables have same size
        2. mlperf: model follows the configuration in MLPerf (v2.1)
        3. rmc1, rmc2, rmc3: models follow the configuration in [DeepRecSys](https://arxiv.org/abs/2001.02772).
    - `{BBB}`: locality of embedding access
        1. zipf
        2. kaggle
        3. uniform
    - `{CCC}`: scaling factor of table size
        - When the value is 1, then the total size of tables is 96 GB.
        - When the value is $x (\not=1)$, then the tables are scaled by $x$.
    - `{DDD}`: batch size
    - `{EEE}`: pooling factor
    - `{FFF}`: training type
        1. sgd: standard SGD training
        2. [dpsgd_b](https://arxiv.org/abs/1607.00133): basic DP-SGD training , referred to as DP-SGD(B)
        3. [dpsgd_r](https://arxiv.org/abs/2009.03106): reweighted DP-SGD training, referred to as DP-SGD(R)
        4. [dpsgd_f](https://arxiv.org/abs/2211.11896): fast-DP-SGD training (same with ghost clipping), referred to as DP-SGD(F). This training is used as our baseline method.
        5. lazydp: our proposal

- In `$PATH_MAIN/result/detailed_latency_breakdown` directory, there are csv files and each file records more detailed latency breakdown result for each training configuration.
    - Comments in `$PATH_MAIN/custom_utils.py` describe the breakdown of this result file.

## Miscellaneous: correctness tests
To verify that the correctness of implementations for DP-SGD(B,R,F) and LazyDP training, run script files whose prefixes are `run_correctness_test_`:
```bash
cd $PATH_MAIN/bench
./run_correctness_test_{NNN}.sh
```
1. `run_correctness_test_1.sh`
    - Compare the results of DP-SGD(B), DP-SGD(R) and DP-SGD(F) with skipping the noise addition.
    - Only check whether the clipping procedure works well.
2. `run_correctness_test_2.sh`
    - Compare the results of SGD and DP-SGD(B) with skipping both the noise addition and gradient clipping.
    - Because we modified the Opacus implementation, we verified that there is no mistake in our modification through this test.

The results of the above correctness tests will be displayed in the terminal.


## Acknowledgement
This research is funded by the generous support from the following organization:
- Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00438851, (SW Starlab) High-performance Privacy-preserving Machine Learning System and System Software)