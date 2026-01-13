import json
import gzip
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import torch

@dataclass
class ProfileEvent:
    name: str
    cat: str
    ph: str
    ts: float
    dur: Optional[float] = None
    pid: int = 0
    tid: int = 0
    args: Optional[Dict[str, Any]] = None

    @property
    def is_cuda_event(self) -> bool:
        return self.cat in ["kernel"]


def load_torch_profile(filename):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        return json.load(f)


def parser_profile(filename):
    profile_data = load_torch_profile(filename)

    if "traceEvents" in profile_data:
        events = profile_data["traceEvents"]
    else:
        raise ValueError("Cannot find traceEvents.")

    events_list = []

    for event in events:
        if "name" not in event:
            continue
        profile_event = ProfileEvent(
            name=event.get("name"),
            cat=event.get("cat", ""),
            ph=event.get("ph", ""),
            ts=event.get("ts", 0.0),
            dur=event.get("dur", None),
            pid=event.get("pid", 0),
            tid=event.get("tid", 0),
            args=event.get("args", {})
        )
        events_list.append(profile_event)

    return events_list


def get_target_ratio(targets, events):
    t_target_events = []
    t_kernel_events = []

    for event in events:
        if not event.is_cuda_event:
            continue
        t_kernel_events.append(event.dur)
        if any(event.name.startswith(prefix) for prefix in targets):
            t_target_events.append(event.dur)

    n_target = len(t_target_events)
    t_target = sum(t_target_events)

    n_kernel = len(t_kernel_events)
    t_kernel = sum(t_kernel_events)

    print(f"n_target_events is {n_target}, n_kernel_events is {n_kernel}")
    print(f"target ratio is {t_target / t_kernel :.3f}")


if __name__ == "__main__":
    targets = [
        "void deep_gemm::sm90_fp8_gemm_1d2d_impl<0u, 4096u, 7168u",
        "void deep_gemm::sm90_fp8_gemm_1d2d_impl<0u, 7168u, 2048u",
    ]

    profile_file = "/sgl-workspace/sglang/experts/profile/profile_batch16_input1024_output128_decode.trace.json.gz"

    events = parser_profile(profile_file)

    get_target_ratio(targets, events)
