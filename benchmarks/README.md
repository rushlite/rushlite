## Benchmarks:

Across the board, my library consistently faster than Pytorch for the CUDA implementations, and many orders of magnitudes slower for the CPU implementation (but I mean who uses CPU for machine learning anyway :p). Would like to note that these benchmarks just look at operators, and Pytorch also has optimizations when dealing with groups of operators, which was not benchmarked here (UPDATE: kernel fusion has been implemented, will add benchmarks in a future PR).

### Measurement protocol

Both harnesses measure synchronized end-to-end per-op time with the same protocol, so the CSVs are directly comparable:

- Inputs (and, for backward, the graph) are built once, outside the timed region.
- Each timed iteration runs a batch of ops followed by one device synchronize, all inside the timed region; the reported time is elapsed / batch size. Without the in-loop synchronize, CUDA benchmarks would only time host-side kernel enqueue.
- Backward is called repeatedly on the retained graph with no reduction on either side: Lamp3's `backward()` seeds `ones_like(out)` internally, and the PyTorch side matches it with `out.backward(torch.ones_like(out), retain_graph=True)`. Gradients accumulate across calls (no zero_grad). A reduction is deliberately not inserted (operator_benchmark's `mean().backward()` is a workaround for PyTorch requiring a scalar); it would dominate the measurement with the reduction's backward rather than the op's own.

Note: results generated before this protocol (the images below) did not synchronize inside the timed region on the Lamp3 side and overstate the CUDA gap. They should be regenerated.

Commands run:

```bash
build/benchmarks/reg_bench_long --benchmark_format=csv --benchmark_time_unit=us > bench_lmp.csv
python -m pt.all_tests --output-csv bench_torch.csv --min-time-per-test 10
```

- GPU: RunPod, NVIDIA RTX 4000 Ada
- CPU: RunPod, 47 GB RAM + 9 vCPU

Docker container: https://github.com/rushlite/rushlite_image

![image](https://github.com/user-attachments/assets/a88e19b3-103e-4e74-ae3e-444d0a35e059)
![image](https://github.com/user-attachments/assets/38618e8a-2aba-449b-a0f7-6a6fe63c32f3)
![image](https://github.com/user-attachments/assets/625b5ac0-6226-43bb-bb3b-02a3ee6b03ac)
