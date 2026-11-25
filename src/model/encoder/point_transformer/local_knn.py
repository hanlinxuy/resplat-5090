import torch
from torch import Tensor


@torch.no_grad()
def local_knn_query(
    k: int,
    points_3d: Tensor,          # [B*V*H*W, 3], scene-major
    extrinsics: Tensor,          # [B, V, 4, 4] or [V, 4, 4]
    intrinsics: Tensor,          # [B, V, 3, 3] or [V, 3, 3], normalized
    v: int, h: int, w: int,
    spatial_radius: int = 3,
    num_neighbor_views: int = 4,
    cross_view_radius: int = 3,
    query_chunk_size: int = 4096,
    fused_candidates: bool = False,
) -> Tensor:
    """Local KNN for structured point clouds from multi-view depth maps.

    Builds a candidate set from:
    1. Per-view 2D spatial neighbors (grid offsets within radius r)
    2. Cross-view neighbors (project to nearest cameras, sample window)

    Then selects top-k nearest from the combined candidate set.

    Return global [B*V*H*W, k] int32 indices in the flattened point array.
    Projection is vectorized over camera pairs within each scene. Only one
    chunk of candidate points is materialized at a time.
    """
    N = v * h * w
    if extrinsics.ndim == 4:
        batch = extrinsics.shape[0]
        if extrinsics.shape != (batch, v, 4, 4) or intrinsics.shape != (batch, v, 3, 3):
            raise ValueError("expected camera tensors with shape [B,V,4,4] and [B,V,3,3]")
        if batch == 0 or points_3d.shape != (batch * N, 3):
            raise ValueError("point array must contain B*V*H*W scene-major points")
        indices = torch.empty((batch * N, k), device=points_3d.device, dtype=torch.int32)
        for scene in range(batch):
            start = scene * N
            indices[start:start + N] = local_knn_query(
                k, points_3d[start:start + N], extrinsics[scene], intrinsics[scene],
                v, h, w, spatial_radius, num_neighbor_views, cross_view_radius,
                query_chunk_size, fused_candidates,
            ).add_(start)
        return indices

    if points_3d.shape != (N, 3) or query_chunk_size <= 0:
        raise ValueError("expected V*H*W points and a positive chunk size")
    device = points_3d.device

    points_grid = points_3d.reshape(v, h, w, 3)

    # --- 1. Camera neighbor list ---
    cam_centers = extrinsics[:, :3, 3]  # [V, 3]
    cam_dist = torch.cdist(cam_centers.unsqueeze(0), cam_centers.unsqueeze(0), p=2)[0]  # [V, V]
    cam_dist_sorted = torch.argsort(cam_dist, dim=1)  # [V, V]
    # Exclude self (index 0 is self), take num_neighbor_views
    nv = min(num_neighbor_views, v - 1)
    neighbor_views = cam_dist_sorted[:, 1:nv + 1]  # [V, nv]

    # --- 2. Spatial candidate indices [N, S] ---
    sr = spatial_radius
    offsets_r = torch.arange(-sr, sr + 1, device=device)
    offsets_c = torch.arange(-sr, sr + 1, device=device)
    dr, dc = torch.meshgrid(offsets_r, offsets_c, indexing="ij")
    dr = dr.reshape(-1)
    dc = dc.reshape(-1)
    # Remove center (0, 0)
    center_mask = (dr != 0) | (dc != 0)
    dr = dr[center_mask]
    dc = dc[center_mask]
    # Candidate offsets for projected points in nearby views.
    cr = cross_view_radius
    cv_offsets_r = torch.arange(-cr, cr + 1, device=device)
    cv_offsets_c = torch.arange(-cr, cr + 1, device=device)
    cv_dr, cv_dc = torch.meshgrid(cv_offsets_r, cv_offsets_c, indexing="ij")
    cv_dr = cv_dr.reshape(-1)  # [C] where C = (2*cr+1)^2
    cv_dc = cv_dc.reshape(-1)
    C = cv_dr.shape[0]

    # Precompute world-to-camera transforms
    w2c = torch.linalg.inv(extrinsics)  # [V, 4, 4]

    # Homogeneous points per view: [V, HW, 4]
    hw = h * w
    ones = torch.ones(v, hw, 1, device=device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_grid.reshape(v, hw, 3), ones], dim=-1)  # [V, HW, 4]

    # Gather target transforms for all (source_view, neighbor) pairs
    w2c_tgt = w2c[neighbor_views]          # [V, nv, 4, 4]
    intr_tgt = intrinsics[neighbor_views]  # [V, nv, 3, 3]

    # Project all source points to all target cameras simultaneously
    # [V, 1, HW, 4] @ [V, nv, 4, 4] -> [V, nv, HW, 4]
    cam_pts = pts_h.unsqueeze(1) @ w2c_tgt.transpose(-1, -2)
    cam_xyz = cam_pts[..., :3]             # [V, nv, HW, 3]
    valid_z = cam_xyz[..., 2] > 0          # [V, nv, HW]

    # Perspective projection + intrinsics
    uv = cam_xyz[..., :2] / cam_xyz[..., 2:3].clamp(min=1e-6)  # [V, nv, HW, 2]
    uv_ones = torch.ones(*uv.shape[:-1], 1, device=device, dtype=uv.dtype)
    uv_h = torch.cat([uv, uv_ones], dim=-1)          # [V, nv, HW, 3]
    proj = uv_h @ intr_tgt.transpose(-1, -2)          # [V, nv, HW, 3]

    # Pixel coords + window offsets
    proj_col = (proj[..., 0] * w - 0.5).round().long()  # [V, nv, HW]
    proj_row = (proj[..., 1] * h - 0.5).round().long()

    proj_row = proj_row.transpose(1, 2).reshape(N, nv)
    proj_col = proj_col.transpose(1, 2).reshape(N, nv)
    valid_z = valid_z.transpose(1, 2).reshape(N, nv)
    target_base = neighbor_views * hw
    candidate_count = dr.numel() + nv * C
    result = torch.empty((N, k), device=device, dtype=torch.int32)
    if fused_candidates and points_3d.is_cuda and points_3d.dtype == torch.float32:
        from .local_knn_candidates import candidate_distances

        points_3d = points_3d.contiguous()
        proj_row, proj_col = proj_row.contiguous(), proj_col.contiguous()
        valid_z, target_base = valid_z.contiguous(), target_base.contiguous()
    else:
        fused_candidates = False
    for start in range(0, N, query_chunk_size):
        end = min(start + query_chunk_size, N)
        query = torch.arange(start, end, device=device)
        if fused_candidates:
            indices, sq_dist = candidate_distances(
                points_3d, proj_row, proj_col, valid_z, target_base,
                start, end, h, w, sr, cr,
            )
        else:
            source_view = query // hw
            row = query % hw // w
            col = query % w
            nb_row = row[:, None] + dr
            nb_col = col[:, None] + dc
            spatial_valid = (nb_row >= 0) & (nb_row < h) & (nb_col >= 0) & (nb_col < w)
            spatial_indices = source_view[:, None] * hw + nb_row.clamp_(0, h - 1) * w + nb_col.clamp_(0, w - 1)

            nb_r = proj_row[start:end, :, None] + cv_dr
            nb_c = proj_col[start:end, :, None] + cv_dc
            cross_valid = ((nb_r >= 0) & (nb_r < h) & (nb_c >= 0) & (nb_c < w)
                           & valid_z[start:end, :, None])
            cross_indices = target_base[source_view, :, None] + nb_r.clamp_(0, h - 1) * w + nb_c.clamp_(0, w - 1)
            indices = torch.cat((spatial_indices, cross_indices.flatten(1)), dim=1)
            valid = torch.cat((spatial_valid, cross_valid.flatten(1)), dim=1)

            candidate_pos = points_3d[indices]
            sq_dist = ((candidate_pos - points_3d[start:end, None]) ** 2).sum(dim=-1)
            sq_dist.masked_fill_(~valid, float("inf"))
        topk_dist, topk_local = sq_dist.topk(min(k, candidate_count), largest=False)
        neighbors = indices.gather(1, topk_local)
        neighbors = torch.where(topk_dist == float("inf"), query[:, None], neighbors)
        result[start:end, :neighbors.shape[1]] = neighbors
        result[start:end, neighbors.shape[1]:] = query[:, None]

    return result
