# Attention Cores

Learning project with Attention cores.

## Structure

- Python refference code for Mutihead Attention
- C++ scalar cores for Mutihead Attention
- C++ SIMD cores for Mutihead Attention, x86_64
- Pybindings
- Pytests for precision check Python refference <-> one of manual implementations
- Pytests with perf check, comparison.

## Current work (branch basic-acceleration-to-refference)

- Using ThreadPool in cores for computation in parallel threads.
- Try to vectorize computation by long axis (seq len) because currently vectorization is by short batch_size axis.
- Searching for other ways to accelerate SIMD version to gain perf closer to refference.

## Global plans

- Build transformer
- Find all places to fuse operations inside transformer.
- Build tiny GPT-like architecture.
- Implement KV-cache for CPU.
- GDN and Sparce Attention, compare compute, memory consumption.
- Loader for ONNX.
- Load and deploy tiny GPT-like model on custom cores, check output.

## Main idea

- First rule: no vibecoding because learning myself process. Long, difficult, many errors, but my errors is my experience.
- Second rule: follow first rule, that's simple :) Open for your recomendations if you're human.