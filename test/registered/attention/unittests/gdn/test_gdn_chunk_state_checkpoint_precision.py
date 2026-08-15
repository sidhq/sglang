"""Regression test: chunked GDN intermediate states (h) are prefix-cache
checkpoints and must round-trip through the recurrent-state pool dtype.

h[c] returned by chunk_gated_delta_rule is the state BEFORE chunk c, i.e. the
state after exactly c*64 tokens; _track_mamba_state_extend scatters a boundary
row into the mamba pool and a later request restores it as its live state.
When the pool is fp32 (mamba_ssm_dtype: float32) but h is allocated in the
bf16 compute dtype, every donated non-chunk-aligned checkpoint is the true
state rounded to bf16 (~2e-3 relative error), which the recurrence then
amplifies on warm restarts.

The reference here is the kernel itself: the state after c*64 tokens is the
in-place fp32 final state of a run truncated to c*64 tokens, so no assumptions
about the recurrence's math conventions are needed.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-large")

CHUNK = 64


class TestGDNChunkStateCheckpointPrecision(CustomTestCase):
    def _make_inputs(self, T, H=4, K=64, V=64, seed=0):
        torch.manual_seed(seed)
        dev = "cuda"
        q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=dev)
        k = torch.nn.functional.normalize(
            torch.randn(1, T, H, K, dtype=torch.bfloat16, device=dev), p=2, dim=-1
        )
        v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device=dev)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=torch.float32, device=dev)
        ).to(torch.bfloat16)
        beta = torch.rand(1, T, H, dtype=torch.bfloat16, device=dev).sigmoid()
        return q, k, v, g, beta

    def _run(self, inputs, T, pool):
        q, k, v, g, beta = (x[:, :T].contiguous() for x in inputs)
        cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        _, _, h = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=pool,
            initial_state_indices=idx,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return h

    def test_h_checkpoints_round_trip_pool_dtype(self):
        T, H, K, V = 200, 4, 64, 64
        inputs = self._make_inputs(T, H, K, V)

        pool_full = torch.zeros(1, H, V, K, dtype=torch.float32, device="cuda")
        h = self._run(inputs, T, pool_full)

        # the fix's contract: checkpoints carry the pool dtype
        self.assertEqual(h.dtype, pool_full.dtype)

        # h[c] must equal the fp32 in-place final state of a run truncated to
        # c*64 tokens (bf16 h storage fails this at ~2e-3 relative error)
        h = h.reshape(-1, H, V, K)
        for c in range(1, T // CHUNK + 1):
            pool_ref = torch.zeros(1, H, V, K, dtype=torch.float32, device="cuda")
            self._run(inputs, c * CHUNK, pool_ref)
            ref = pool_ref[0]
            got = h[c].to(torch.float64)
            rel = (got - ref.double()).norm() / (ref.double().norm() + 1e-12)
            self.assertLess(
                rel.item(),
                1e-5,
                f"h[{c}] (state after {c * CHUNK} tokens) deviates from the fp32 "
                f"reference by rel={rel.item():.3e}; intermediate-state "
                f"checkpoints must not be quantized below the pool dtype",
            )

    def test_bf16_pool_unchanged(self):
        # with a bf16 pool the checkpoints stay bf16 (no behavior change)
        T, H, K, V = 128, 2, 64, 64
        inputs = self._make_inputs(T, H, K, V, seed=1)
        pool = torch.zeros(1, H, V, K, dtype=torch.bfloat16, device="cuda")
        h = self._run(inputs, T, pool)
        self.assertEqual(h.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
