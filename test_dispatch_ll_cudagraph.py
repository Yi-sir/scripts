import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back

def do_sth_hook(hook):
    # do sth to simulate moe gemm
    hook()

def generate_deepep_inputs(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                           rank: int, num_ranks: int, buffer: deep_ep.Buffer, rank_offset: int = 128,
                           return_recv_hook = True, use_logfmt: bool = False):
    num_local_experts = num_experts // num_ranks

    current_x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * 0.1

    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device="cuda")

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda").abs()

    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1


    recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
    do_sth_hook(hook) if return_recv_hook else None
    packed_recv_x = (recv_x[0], recv_x[1].contiguous())
    simulated_gemm_x = per_token_cast_back(packed_recv_x[0].view(-1, hidden), packed_recv_x[1].view(-1, hidden // 128)).view(packed_recv_x[0].shape)

    combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                            use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
    do_sth_hook(hook) if return_recv_hook else None
    torch.cuda.synchronize()

    return current_x, topk_idx, cumulative_local_expert_recv_stats, topk_weights, simulated_gemm_x


def test_func(buffer, current_x, topk_idx, num_tokens, num_experts, cumulative_local_expert_recv_stats,
              return_recv_hook, simulated_gemm_x, topk_weights, use_logfmt):
    recv_x, recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                    use_fp8=True, async_finish=False, return_recv_hook=return_recv_hook)
    do_sth_hook(hook) if return_recv_hook else None
    combined_x, event, hook = buffer.low_latency_combine(simulated_gemm_x, topk_idx, topk_weights, handle,
                                                            use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
    do_sth_hook(hook) if return_recv_hook else None


def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
         rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
         use_logfmt: bool = False, seed: int = 0, capture_iters: int = 10):
    print(f"[rank {rank}] test_main...")

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    return_recv_hook = True

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    def prepare_cuda_graph():
        print(f"[rank {rank}] warmup...")
        current_x, topk_idx, cumulative_local_expert_recv_stats, topk_weights, simulated_gemm_x = generate_deepep_inputs(
            num_tokens, hidden, num_experts, num_topk, rank, num_ranks, buffer, return_recv_hook=return_recv_hook, use_logfmt=use_logfmt)
        print(f"[rank {rank}] warmup passed")

        print(f"[rank {rank}] Capturing CUDA Graph...")
        capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(capture_stream):
            graph.capture_begin()
            test_func(buffer, current_x, topk_idx, num_tokens, num_experts, cumulative_local_expert_recv_stats,
                return_recv_hook, simulated_gemm_x, topk_weights, use_logfmt)
            graph.capture_end()
        torch.cuda.synchronize()
        print(f"[rank {rank}] CUDA Graph captured successfully!")

        return graph, (current_x, topk_idx, cumulative_local_expert_recv_stats, topk_weights, simulated_gemm_x)

    graph1, graph1_inputs = prepare_cuda_graph()
    graph2, graph2_inputs = prepare_cuda_graph()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.ProfilerActivity.CPU
        ]
    ) as prof:
        for _ in range(capture_iters):
            graph1.replay()
            cache.zero_()
            graph2.replay()
        torch.cuda.synchronize()

    trace_path = f"./profile/2_addr_graph/{rank}_deepep_graph_2.trace.json.gz"

    if not os.path.exists(trace_path):
        prof.export_chrome_trace(trace_path)
        print(f"[ranks {rank}] trace file saved")


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
              use_logfmt=args.use_logfmt, seed=1, capture_iters=10)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256,
                       help='Number of experts (default: 267)')
    parser.add_argument('--allow-mnnvl', action="store_true",
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
