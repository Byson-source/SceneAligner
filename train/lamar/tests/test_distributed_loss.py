"""distributed_feature_matching_loss on 2 CPU processes (gloo) against feature_matching_loss on the concatenated
batch: the same loss and gradients (no GPU, no LaMAR data).

    python -m pytest train/lamar/tests
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")  # scenealigner.training imports the inference dependencies
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from scenealigner.training import distributed_feature_matching_loss, feature_matching_loss  # noqa: E402

WORLD, B, N, C = 2, 3, 5, 8


def batch(negatives):
    """Features (W * B, N, C) of the global batch, its valid points (as many on each GPU), same_scene (W * B, W * B)
    and the arguments near / soft of the loss: random symmetric pairs of points (not a point and itself) and
    symmetric weights of soft positives (1 for a point and itself)."""
    gen = torch.Generator().manual_seed(0)
    feat_a, feat_b = torch.randn(WORLD * B, N, C, generator=gen), torch.randn(WORLD * B, N, C, generator=gen)
    valid = torch.ones(WORLD * B, N, dtype=torch.bool)
    valid[[0, B], -1] = False  # one padded point per GPU
    scene = torch.tensor([0, 1, 0, 1, 1, 2])
    same_scene = (scene[:, None] == scene[None]) & ~torch.eye(WORLD * B, dtype=torch.bool)
    near = torch.rand(WORLD * B * N, WORLD * B * N, generator=gen) < 0.2
    eye = torch.eye(WORLD * B * N, dtype=torch.bool)
    near = (near | near.T) & ~eye
    soft = torch.rand(WORLD * B * N, WORLD * B * N, generator=gen) * near
    pairs = {"none": {}, "near": {"near": near}, "soft": {"soft": (soft + soft.T) / 2 + eye}}
    return feat_a, feat_b, valid, same_scene if negatives == "scene" else torch.zeros_like(same_scene), pairs


def worker(rank, port, negatives, mode, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    feat_a, feat_b, valid, same_scene, pairs = batch(negatives)
    local = slice(rank * B, (rank + 1) * B)
    a, b = feat_a[local].clone().requires_grad_(), feat_b[local].clone().requires_grad_()
    rows = {k: v[rank * B * N:(rank + 1) * B * N] for k, v in pairs[mode].items()}
    loss = distributed_feature_matching_loss(a, b, same_scene[local], valid[local], **rows)
    loss.backward()
    out[rank] = (loss.detach(), a.grad / WORLD, b.grad / WORLD)  # DDP averages the gradients over the GPUs
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["none", "near", "soft"])
@pytest.mark.parametrize("negatives", ["scene", "batch"])
def test_distributed_loss_matches_the_concatenated_batch(negatives, mode):
    out = mp.Manager().dict()
    port = 29500 + (negatives == "batch") + 2 * ["none", "near", "soft"].index(mode)
    mp.spawn(worker, args=(port, negatives, mode, out), nprocs=WORLD)
    feat_a, feat_b, valid, same_scene, pairs = batch(negatives)
    feat_a.requires_grad_(), feat_b.requires_grad_()
    loss = feature_matching_loss(feat_a, feat_b, same_scene, valid, **pairs[mode])
    loss.backward()
    assert torch.allclose(sum(out[r][0] for r in range(WORLD)) / WORLD, loss.detach(), atol=1e-5)
    assert torch.allclose(torch.cat([out[r][1] for r in range(WORLD)]), feat_a.grad, atol=1e-6)
    assert torch.allclose(torch.cat([out[r][2] for r in range(WORLD)]), feat_b.grad, atol=1e-6)
