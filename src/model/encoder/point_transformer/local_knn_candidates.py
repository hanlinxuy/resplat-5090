"""Fuse candidate coordinates and distances for image-grid local KNN."""

import torch
import triton
import triton.language as tl


@triton.jit
def _candidate_distances(Points, ProjRow, ProjCol, ValidZ, TargetBase, Indices, Distances,
                         START, END, H: tl.constexpr, W: tl.constexpr, NV: tl.constexpr,
                         SR: tl.constexpr, CR: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
                         CANDIDATES: tl.constexpr, COLS: tl.constexpr, ROWS: tl.constexpr):
    row = START + tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cand = tl.arange(0, COLS)
    spatial = cand < S
    spatial_offset = cand + (cand >= S // 2).to(tl.int32)
    sr_diameter = 2 * SR + 1
    image_row = row % (H * W) // W
    image_col = row % W
    view = row // (H * W)
    nr = image_row[:, None] + spatial_offset[None, :] // sr_diameter - SR
    nc = image_col[:, None] + spatial_offset[None, :] % sr_diameter - SR
    view_base = view[:, None] * (H * W) + tl.full((1, COLS), 0, tl.int32)
    positive_z = tl.full((ROWS, COLS), True, tl.int1)

    if NV:
        cv_diameter = 2 * CR + 1
        view_slot = (cand - S) // C
        window_offset = (cand - S) % C
        proj_index = row[:, None] * NV + view_slot[None, :]
        cross = (row[:, None] < END) & (cand[None, :] >= S) & (cand[None, :] < CANDIDATES)
        projected_row = tl.load(ProjRow + proj_index, cross, other=0)
        projected_col = tl.load(ProjCol + proj_index, cross, other=0)
        z = tl.load(ValidZ + proj_index, cross, other=False)
        target = tl.load(TargetBase + view[:, None] * NV + view_slot[None, :], cross, other=0)
        nr = tl.where(spatial[None, :], nr, projected_row + window_offset[None, :] // cv_diameter - CR)
        nc = tl.where(spatial[None, :], nc, projected_col + window_offset[None, :] % cv_diameter - CR)
        view_base = tl.where(spatial[None, :], view_base, target)
        positive_z = spatial[None, :] | z

    valid = ((row[:, None] < END) & (cand[None, :] < CANDIDATES) & positive_z
             & (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W))
    neighbor = (view_base + nr * W + nc).to(tl.int32)
    qx = tl.load(Points + row * 3, row < END, other=0).to(tl.float32)
    qy = tl.load(Points + row * 3 + 1, row < END, other=0).to(tl.float32)
    qz = tl.load(Points + row * 3 + 2, row < END, other=0).to(tl.float32)
    dx = tl.load(Points + neighbor * 3, valid, other=0).to(tl.float32) - qx[:, None]
    dy = tl.load(Points + neighbor * 3 + 1, valid, other=0).to(tl.float32) - qy[:, None]
    dz = tl.load(Points + neighbor * 3 + 2, valid, other=0).to(tl.float32) - qz[:, None]
    distance = (dx * dx + dy * dy) + dz * dz
    offset = (row[:, None] - START) * CANDIDATES + cand[None, :]
    mask = (row[:, None] < END) & (cand[None, :] < CANDIDATES)
    tl.store(Indices + offset, tl.where(valid, neighbor, row[:, None]), mask)
    tl.store(Distances + offset, tl.where(valid, distance, float("inf")), mask)


def candidate_distances(points, proj_row, proj_col, valid_z, target_base, start, end,
                        h, w, spatial_radius, cross_radius):
    nv = proj_row.shape[1]
    s = (2 * spatial_radius + 1) ** 2 - 1
    c = (2 * cross_radius + 1) ** 2
    count = s + nv * c
    indices = torch.empty((end - start, count), device=points.device, dtype=torch.int32)
    distances = points.new_empty((end - start, count))
    if count:
        rows = 4
        _candidate_distances[(triton.cdiv(end - start, rows),)](
            points, proj_row, proj_col, valid_z, target_base, indices, distances, start, end,
            h, w, nv, spatial_radius, cross_radius, s, c, count, triton.next_power_of_2(count), rows,
            num_warps=4, enable_fp_fusion=False,
        )
    return indices, distances
