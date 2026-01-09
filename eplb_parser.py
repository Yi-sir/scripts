from pathlib import Path
import itertools
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from sglang.srt.eplb.eplb_algorithms import rebalance_experts, compute_algorithm

NUM_DENSE_LAYERS = 3
N_GROUPS = 8
N_GPUS_PER_GROUP = 8
NUM_ROUTED_EXPERTS = 256

log_ctrl = False
print = print if log_ctrl else (lambda *args, **kwargs: None)

def draw_heat_map(weight):
    assert len(weight.shape) == 3   # [1000, 61, 256] [rebalance_interval, layers, logical_experts]

    weight = weight.sum(dim=0)

    num_layers, num_experts = weight.shape
    heatmap_data = weight.numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(label='Expert Heat')

    plt.xlabel('Expert ID')
    plt.ylabel('Layer ID')

    plt.savefig("./imgs/expert_heatmap.png")


def parse_experts_heat(path):
    if path.endswith(".pt"):
        data = torch.load(path, weights_only=True)
        weight = data["logical_count"].cpu()
    elif path.endswith(".json"):
        data_dict = json.loads(Path(path).read_text())
        weight = torch.tensor(data_dict["logical_count"]).unsqueeze(0)
    # draw_heat_map(weight)
    return weight

def get_real_physical_weight(expert_weight_layer, physical_expert_map_layer):
    n_logical_experts = expert_weight_layer.shape[0]
    n_physical_experts = physical_expert_map_layer.shape[0]

    count_per_logical = torch.zeros(n_logical_experts, dtype=torch.float)
    for i in range(n_physical_experts):
        logical_idx = physical_expert_map_layer[i]
        count_per_logical[logical_idx] += 1

    physical_weights = torch.zeros(n_physical_experts)
    for i in range(n_physical_experts):
        logical_idx = physical_expert_map_layer[i]
        if count_per_logical[logical_idx] > 0:
            physical_weights[i] = expert_weight_layer[logical_idx] / count_per_logical[logical_idx]

    return physical_weights

def draw_ratio(ratio_origin, ratio_eplb, ratio_weight, n_nodes, n_red_experts):
    plt.figure(figsize=(10, 6))
    num_layers = len(ratio_origin)
    x = np.arange(0, num_layers, 1)
    if ratio_origin is not None:
        plt.plot(x, ratio_origin, label='origin', color='blue', linewidth=2, marker='o', markersize=4)
    if ratio_eplb is not None:
        plt.plot(x, ratio_eplb, label='eplb', color='red', linewidth=2, linestyle='--', marker='s', markersize=4)
    if ratio_weight is not None:
        plt.plot(x, ratio_weight, label='gpu weight', color='green', linewidth=2, linestyle=':', marker='s', markersize=4)

    plt.title('expert balancedness/gpu weight', fontsize=14, fontweight='bold')
    plt.xlabel('layer_id', fontsize=12)
    plt.ylabel('balancedness', fontsize=12)

    plt.legend(fontsize=12)

    plt.savefig(f"./imgs/sglang_dataset/expert_balancedness_EP{n_nodes * N_GPUS_PER_GROUP}_256_{n_red_experts}.png")

def rebalance(logical_count, n_nodes, n_redundant_experts):
    alg = compute_algorithm("auto", N_GROUPS, n_nodes)
    num_physical_experts = NUM_ROUTED_EXPERTS + n_redundant_experts
    ep_size = n_nodes * N_GPUS_PER_GROUP

    physical_to_logical_map, _, _ = (
        rebalance_experts(
            tokens_per_expert=logical_count,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_physical_experts // ep_size,
            num_groups=N_GROUPS,
            num_nodes=n_nodes,
            algorithm=alg
        )
    )

    return physical_to_logical_map

def parse_experts_dist(logical_count):
    """
    给定一个原始专家流量，在不同条件下计算EPLB之后的专家分布，比较EPLB前后每一层的专家平均流量除以专家最大流量
    """
    expert_weight = logical_count.sum(dim=0)
    _, num_logical_experts = expert_weight.shape

    # file_path = Path("EP0.npy") # EP几都一样
    # experts_dist = np.load(file_path) # physical to logical

    all_combinations = [(1, 0), (1, 8), (1, 16), (1, 32),
                        (2, 0), (2, 16), (2, 32),
                        (4, 0), (4, 32)]

    for (n_node, n_red_expert) in all_combinations:

        experts_dist = rebalance(logical_count, n_node, n_red_expert)
        ep_rank = n_node * N_GPUS_PER_GROUP
        num_experts_per_ep_rank = (NUM_ROUTED_EXPERTS + n_red_expert) // ep_rank

        num_layers, num_physical_experts = experts_dist.shape

        experts_per_gpu = num_logical_experts // ep_rank
        physical_experts_per_gpu = num_physical_experts // ep_rank

        origin_ratio_arr = np.zeros(num_layers)
        eplb_ratio_arr = np.zeros(num_layers)
        max_weight_ratio_arr = np.zeros(num_layers)

        for layer_id in range(NUM_DENSE_LAYERS, num_layers):
            print(f"layer id: {layer_id}")

            # eplb之前，也就是只考虑EPx，不考虑冗余专家
            expert_weight_layer = expert_weight[layer_id]
            # 流量分配到每个ep rank上，计算每张卡的总流量
            grouped = expert_weight_layer.view(ep_rank, -1)
            # 单卡流量的最大值
            max_sum_on_gpu = grouped.sum(dim=1).max()
            # 再计算每张卡的平均流量
            mean = grouped.mean(dtype=torch.float32).item() * (NUM_ROUTED_EXPERTS // ep_rank)
            print(f"\torigin max heat is {max_sum_on_gpu}")
            print(f"\torigin mean heat is {mean}")
            r = mean/max_sum_on_gpu
            print(f"\tmean/max = {r:.3f}")
            origin_ratio_arr[layer_id] = r

            # eplb之后
            physical_expert_map_layer = experts_dist[layer_id]
            weight_after_eplb_layer = get_real_physical_weight(expert_weight_layer, physical_expert_map_layer)
            grouped_eplb = weight_after_eplb_layer.view(ep_rank, -1)
            max_sum_on_gpu_eplb = grouped_eplb.sum(dim=1).max()
            mean = grouped_eplb.mean(dtype=torch.float32).item() * num_experts_per_ep_rank
            print(f"\teplb max heat is {max_sum_on_gpu_eplb}")
            print(f"\torigin mean heat is {mean}")
            rr = mean/max_sum_on_gpu_eplb
            print(f"\tmean/max = {rr:.3f}")
            eplb_ratio_arr[layer_id] = rr

            # eplb后/前，1 - 单卡最大流量的比值，代表优化幅度
            max_weight_ratio = 1 - max_sum_on_gpu_eplb / max_sum_on_gpu
            max_weight_ratio_arr[layer_id] = max_weight_ratio

        draw_ratio(origin_ratio_arr, eplb_ratio_arr, None, n_node, n_red_expert)


if __name__ == "__main__":
    # /sgl-workspace/sglang/experts/weight/attachment_ep_statistics/prefill_in1024.json
    # /sgl-workspace/sglang/experts/weight/sharegpt_pd_mix.pt
    # /sgl-workspace/sglang/experts/weight/110_decode.pt
    logical_count = parse_experts_heat("/sgl-workspace/sglang/experts/weight/attachment_ep_statistics/prefill_in1024.json")
    parse_experts_dist(logical_count)
