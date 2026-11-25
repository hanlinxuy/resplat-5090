"""Opt-in steady-state timing for the original single-GPU Lightning training loop."""

import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from pytorch_lightning.callbacks import Callback

from .profile_ranges import profile_backend, profile_range


class SingleGPUProfile(Callback):
    def __init__(self, path, warmup, samples, *, capture=False, torch_trace=None):
        if warmup < 0 or samples < 1:
            raise ValueError("profile warmup must be nonnegative and samples must be positive")
        self.path = Path(path)
        self.warmup = warmup
        self.samples = samples
        self.capture = capture
        self.rows = []
        self.events = None
        self.previous_end = None
        self.started = False
        self.finished = False
        self.torch_trace = torch_trace
        self.backend = "nsys" if capture else "torch" if torch_trace else None
        if self.capture and self.torch_trace:
            raise ValueError("run Nsight and PyTorch Profiler in separate processes")

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != 1 or pl_module.device.type != "cuda":
            raise ValueError("ReSplat baseline profiling requires one CUDA GPU")
        self.first_step = trainer.global_step
        self.hooks = []
        self.module_ranges = []
        if self.backend:
            for name, module in pl_module.named_modules():
                if name in ("encoder.depth_predictor", "encoder.gaussian_regressor", "encoder.pt",
                            "encoder.gaussian_head", "decoder") or (
                            name.startswith("losses.") and name.count(".") == 1) or (
                            name.startswith("encoder.pt.blocks.") and name.endswith(".attn")) or (
                            name.startswith("encoder.pt.mv_blocks.") and name.endswith(".conv")):
                    self.hooks.append(module.register_forward_pre_hook(
                        lambda _module, _args, label=name: self._push(label),
                    ))
                    self.hooks.append(module.register_forward_hook(
                        lambda _module, _args, _output: self._pop(), always_call=True,
                    ))

    def _push(self, name):
        if self.events is not None:
            scope = profile_range(name)
            scope.__enter__()
            self.module_ranges.append(scope)

    def _pop(self):
        if self.events is not None:
            self.module_ranges.pop().__exit__(None, None, None)

    def _next_phase(self, name):
        self.phase.__exit__(None, None, None)
        self.phase = profile_range(name)
        self.phase.__enter__()

    def on_train_end(self, trainer, pl_module):
        if self.rows and not self.finished:
            self._finish(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        for hook in self.hooks:
            hook.remove()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        index = trainer.global_step - self.first_step
        if not self.warmup <= index < self.warmup + self.samples:
            return
        if not self.started:
            self.started = True
            torch.cuda.synchronize()
            self.startup_peak_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            if self.capture:
                torch.cuda.profiler.start()
            if self.torch_trace:
                self.torch_profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, with_flops=True,
                )
                self.torch_profiler.start()
        self.batch_start = perf_counter()
        self.update_index = trainer.global_step + 1
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.events[0].record()
        self.backend_token = profile_backend.set(self.backend)
        self.phase = profile_range("train_forward_loss")
        self.phase.__enter__()

    def on_before_backward(self, trainer, pl_module, loss):
        if self.events:
            self.events[1].record()
            self._next_phase("train_backward")

    def on_after_backward(self, trainer, pl_module):
        if self.events:
            self.events[2].record()
            self._next_phase("train_update")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.events:
            return
        self.events[3].record()
        self.phase.__exit__(None, None, None)
        profile_backend.reset(self.backend_token)
        self.events[3].synchronize()
        end = perf_counter()
        a, b, c, d = self.events
        loss = None if outputs is None else float(outputs["loss"].detach().item())
        if loss is not None and not math.isfinite(loss):
            raise FloatingPointError(f"non-finite loss in update {self.update_index}")
        # Lightning reports the loss divided by accumulation steps.
        if loss is not None:
            loss *= trainer.accumulate_grad_batches
        self.rows.append(dict(
            step=trainer.global_step, update=self.update_index, scenes=int(batch["context"]["image"].shape[0]),
            forward_loss_ms=a.elapsed_time(b), backward_ms=b.elapsed_time(c), update_ms=c.elapsed_time(d),
            cuda_step_ms=a.elapsed_time(d), wall_batch_ms=(end - self.batch_start) * 1e3,
            wall_cadence_ms=None if self.previous_end is None else (end - self.previous_end) * 1e3,
            loss=loss, refinement_iterations=getattr(
                getattr(pl_module, "encoder", None), "profile_refinement_iterations", None),
            model_shape=getattr(pl_module, "profile_training_shape", None),
            scene_keys=[str(value) for value in batch.get("scene", ())],
            context_views=batch["context"]["index"].detach().cpu().tolist() if "index" in batch["context"] else None,
        ))
        self.previous_end = end
        self.events = None
        if trainer.global_step == self.first_step + self.warmup + self.samples:
            self._finish(trainer, pl_module)

    def _finish(self, trainer, pl_module):
        self.finished = True
        if self.capture:
            torch.cuda.profiler.stop()
        if self.torch_trace:
            self.torch_profiler.stop()
            trace = Path(self.torch_trace)
            trace.parent.mkdir(parents=True, exist_ok=True)
            self.torch_profiler.export_chrome_trace(str(trace))
            trace.with_suffix(".txt").write_text(self.torch_profiler.key_averages(
                group_by_input_shape=True,
            ).table(sort_by="self_device_time_total", row_limit=80))
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        audit = {}
        if os.environ.get("RESPLAT_PROFILE_PARAMETER_AUDIT") == "1":
            optimizer = trainer.optimizers[0]
            for name, param in pl_module.named_parameters():
                if name not in (
                    "encoder.pt.blocks.0.attn.qkv.weight", "encoder.pt.mv_blocks.0.conv.weight",
                    "encoder.gaussian_head.0.weight", "encoder.update_proj.weight",
                ):
                    continue
                state = optimizer.state.get(param, {})
                audit[name] = {}
                for key, value in (
                    ("parameter", param), ("grad", param.grad),
                    ("exp_avg", state.get("exp_avg")), ("exp_avg_sq", state.get("exp_avg_sq")),
                ):
                    if value is not None:
                        flat = value.detach().flatten()
                        sample = flat[::max(1, flat.numel() // 256)][:256].float().cpu()
                        if not torch.isfinite(sample).all():
                            raise FloatingPointError(f"non-finite sampled {key} for {name}")
                        audit[name][key] = sample.tolist()
        properties = torch.cuda.get_device_properties(pl_module.device)
        medians = {
            key: median(row[key] for row in self.rows if row[key] is not None)
            for key in self.rows[0] if key.endswith("_ms") and any(row[key] is not None for row in self.rows)
        }
        updates = {}
        for row in self.rows:
            update = updates.setdefault(row["update"], dict(microbatches=0, scenes=0, cuda_ms=0.))
            update["microbatches"] += 1
            update["scenes"] += row["scenes"]
            update["cuda_ms"] += row["cuda_step_ms"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(
            profile_mode=self.backend or "timing",
            capture_complete=trainer.global_step == self.first_step + self.warmup + self.samples,
            gpu=properties.name, capability=torch.cuda.get_device_capability(pl_module.device),
            sm_count=properties.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            fp32_matmul_precision=torch.get_float32_matmul_precision(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            synthetic_random_weights=os.environ.get("RESPLAT_SYNTHETIC_RANDOM_WEIGHTS") == "1",
            perceptual_bf16=os.environ.get("RESPLAT_PERCEPTUAL_BF16") == "1",
            perceptual_channels_last=os.environ.get("RESPLAT_PERCEPTUAL_CHANNELS_LAST") == "1",
            perceptual_checkpoint=os.environ.get("RESPLAT_PERCEPTUAL_CHECKPOINT") == "1",
            perceptual_view_chunk=int(os.environ.get("RESPLAT_PERCEPTUAL_VIEW_CHUNK", "0")),
            depth_candidates=pl_module.encoder.cfg.num_depth_candidates,
            depth_cost_chunk=pl_module.encoder.cfg.cost_volume_depth_chunk,
            depth_activation_checkpointing=pl_module.encoder.cfg.depth_activation_checkpointing,
            point_blocks=pl_module.encoder.cfg.num_blocks,
            local_knn=pl_module.encoder.cfg.init_use_local_knn,
            fused_knn_candidates=pl_module.encoder.cfg.init_fused_knn_candidates,
            fused_local_attention_rows=pl_module.encoder.cfg.init_fused_local_attention_rows,
            channels_last_conv=pl_module.encoder.cfg.init_channels_last_conv,
            point_activation_checkpointing=pl_module.encoder.cfg.init_use_checkpointing,
            gaussians_per_latent=pl_module.encoder.cfg.init_gaussian_multiple,
            cuda_allocator_limit_gib=os.environ.get("RESPLAT_CUDA_ALLOCATOR_LIMIT_GIB"),
            fused_adamw=bool(trainer.optimizers[0].defaults.get("fused")),
            total_params=sum(p.numel() for p in pl_module.parameters()),
            trainable_params=sum(p.numel() for p in pl_module.parameters() if p.requires_grad),
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            startup_peak_reserved_bytes=self.startup_peak_reserved,
            parameter_audit=audit,
            accumulation=trainer.accumulate_grad_batches, updates=updates, median=medians, samples=self.rows,
        ), indent=2) + "\n")
