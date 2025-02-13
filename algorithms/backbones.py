import torch
import torch.nn as nn

from algorithms import blocks
from algorithms.base import ExtendedModule
from algorithms.utils import PositionEncoding
from algorithms.gilbert_xy2d import get_gilbert_lut
from algorithms.counting import CountedMatmul
from math import sqrt, prod
import torch.nn.functional as func
from algorithms.modules import TokenBuffer


def get_heatmap_offset_indices(input_size, max_local_offset):
    orig_h, orig_w = input_size
    orig_inds_map = torch.arange(orig_h * orig_w).reshape(orig_h, orig_w)
    padded_h = orig_h + 2 * max_local_offset
    padded_w = orig_w + 2 * max_local_offset
    padded_inds_map = torch.arange(padded_h * padded_w).reshape(padded_h, padded_w)
    orig_inds_map_zero_bounds = torch.zeros_like(padded_inds_map)
    orig_inds_map_zero_bounds[max_local_offset:max_local_offset+orig_h, max_local_offset:max_local_offset+orig_w] =\
        orig_inds_map
    n_offsets = (2 * max_local_offset + 1) ** 2
    local_offset_inds = torch.zeros(orig_h, orig_w, n_offsets, dtype=torch.int64)
    offset_ind = -1
    copy_inds_map = torch.zeros_like(local_offset_inds)
    for y_offset in range(-max_local_offset, max_local_offset + 1):
        for x_offset in range(-max_local_offset, max_local_offset + 1):
            offset_ind += 1
            top_in_padded = max_local_offset + y_offset
            left_in_padded = max_local_offset + x_offset
            padded_inds_for_offset = padded_inds_map[top_in_padded:top_in_padded + orig_h,
                                     left_in_padded:left_in_padded + orig_w]
            local_offset_inds[:, :, offset_ind] = padded_inds_for_offset
            copy_inds_for_offset = orig_inds_map_zero_bounds[top_in_padded:top_in_padded + orig_h,
                                   left_in_padded:left_in_padded + orig_w]
            copy_inds_map[:, :, offset_ind] = copy_inds_for_offset
    copy_inds_lut = torch.flatten(copy_inds_map)
    return local_offset_inds, copy_inds_lut


def _reduce_rollout_mat_after_pruning(result, kept_inds):
    result_after_y_samp = torch.gather(result, dim=1, index=kept_inds[..., None].expand(-1, -1, result.shape[-1]))
    result = torch.gather(result_after_y_samp, dim=2,
                          index=kept_inds[:, None, ...].expand(-1, result_after_y_samp.shape[1], -1))
    return result


class ViTBackbone(ExtendedModule):
    """
    Common backbone for vision Transformers.
    """

    def __init__(
            self,
            block_config,
            depth,
            position_encoding_size,
            input_size,
            block_class="Block",
            has_class_token=False,
            window_indices=(),
            windowed_class=None,
            windowed_overrides=None,
            token_drop=False,
            drop_block_indices=(),
            num_to_drop_in_block=(),
            first_drop_pattern="odd_even",
            late_drop_pattern="odd_even",
            drop_using_rollout=False,
            force_compute_rollout=False,
            rollout_block_indices=(),
            rollout_head_fusion="mean",
            rollout_final_norm="sum",
            first_frame_prune_disable=False,
            rollout_keep_ratio=1.0,
            rollout_fg_samples_ratio=1.0,
            rollout_bg_samples_ratio=1.0,
            rollout_reset_frq=0,
            rollout_spatial_interp=False,
            first_block_class=None,
            last_block_class=None,
            output_buffer=False,
            copy_output_dbg_mode=False,
            reset_full_process=False,
            rollout_by_bbox=False,
            rollout_bbox_score_thr=0.1,
            rollout_only_global=True,
            rollout_pool_size=None,
            max_local_offset=0,
            offset_relative_prct_thr=None,
            offset_using_patch_embed=False,
            rollout_gilbert_sampling=False,
            propagate_rollout=False,
            maskvd_prune=False,
            maskvd_bbox_score_thr=0.5,
            maskvd_gilbert_sampling=False,
            maskvd_add_noise=False,
            rollout_backward=False,
    ):
        """
        :param block_config: A dict containing kwargs for the
        block_class constructor
        :param depth: The number of blocks to use
        :param position_encoding_size: The size (in tokens) assumed for
        position encodings
        :param input_size: The expected size of the inputs in tokens
        :param block_class: The specific Block class to use (see
        blocks.py for options)
        :param has_class_token: Whether to add an extra class token
        :param window_indices: Block indices that should use windowed
        attention
        :param windowed_class: The specific Block class to use with
        windowed attention (if None, fall back to block_class)
        :param windowed_overrides: A dict containing kwargs overrides
        for windowed_class
        :param token_drop: indicates that token drop mode is enabled
        :param drop_block_indices: Block indices after which token drop
        takes place
        :param num_to_drop_in_block: number of tokens to drop per block
        - this corresponds to the blocks in "drop_block_indices"
        :param first_drop_pattern: the tokens groups pattern to use
        in the first pruning stage (default: naive odd-even)
        :param late_drop_pattern: the tokens groups pattern to use
        after the first pruning stage (default: naive odd-even)
        :param drop_using_rollout: Use attention rollout in token
        drop decision
        :param force_compute_rollout: calculate attention rollout
        even if not dropping or not using rollout in decision
        :param rollout_block_indices: block indices in which rollout
        should be used in token drop decision
        :param rollout_head_fusion: method to combine attention from
         all heads during rollout computation
        :param rollout_final_norm: normalization method for the final
        rollout map (default: normalize by sum)
        :param first_frame_prune_disable: disable the spatial based
        pruning on the first frame
        :param rollout_keep_ratio: the ratio of tokens to keep using
        the rollout information ("foreground tokens")
        :param rollout_fg_samples_ratio: the ratio of tokens to sample
        from the "foreground region"
        :param rollout_bg_samples_ratio: the ratio of tokens to sample
        from the "background region"
        :param rollout_reset_frq: if set as integer value k > 0, every
        k frames the rollout map from previous frame will not be used
        and pruning will be done using current frame information only
        :param rollout_spatial_interp: if set to true, the rollout map
        for pruned tokens (in a frame where pruning is based on rollout)
        will be interpolated using "semantic" distance in current frame.
        Note that in the "reset frames" this is always the case. if set
        to false, rollout is interpolated from the previous frame at the
        same position.
        :param first_block_class: option to force the class of the first
        transformer block
        :param last_block_class: option to force the class of the last
        transformer block
        :param output_buffer: maintain a buffer of the backbone output (in
        full resolution) to allow forwarding features to detection head
        :param copy_output_dbg_mode: debug mode in which the output
        feature map from the last fully processed frame is simply copied
        to the next frames (until next reset). This bypasses the pruning
        operation in early blocks.
        :param reset_full_process: perform full processing at "rollout
        reset frames"
        :param rollout_by_bbox: use predicted bounding boxes in order
         to determine anchor tokens for attention rollout computation.
        :param rollout_bbox_score_thr: score threshold to use for
        determining which bounding boxes to use for attention rollout
        anchors mask generation.
        :param rollout_only_global: compute attention rollout based only
        on the transformer blocks with global attention. This is an
        approximated computation for architectures with "windowed"
        attention blcoks
        :param rollout_pool_size: pooling size to apply to the attention
        matrices for computation saving (supported values: 2,4,None).
        By default set to None (no pooling)
        :param max_local_offset: if larger than 0, the rollout value
        at frame t+1 will be taken from the most similar token at a
        (2*max_local_offset+1) * (2*max_local_offset+1) neighborhood
        in frame t
        :param offset_relative_prct_thr: positive value indicating
        the minimal percentage increase compared to "keep in same place"
        needed to allow applying offset for the heatmap at the given token
        :param offset_using_patch_embed: compute the distance for offset
        calculation on the patch embedding before first layer
        :param rollout_gilbert_sampling: sample tokens from foreground
        and background regions defined by rollout using gilbert
        :param propagate_rollout: do gradual drop per layer using
        input rollout map propagation in current frame's layers
        :param maskvd_prune: perform pruning similar to maskVD based
        on previous frame's bounding boxes
        param maskvd_bbox_score_thr: the score threshold to use
        when applying maskVD pruning
        param maskvd_gilbert_sampling: sample tokens from foreground
        and background regions defined by bboxes using gilbert
        param maskvd_add_noise: add noise to mask_vd binary mask in
        order to remove spatial artifacts
        param rollout_backward: compute rollout from the last attention
        matrix backward using matrix-vector multiplications
        """
        super().__init__()
        self.position_encoding = PositionEncoding(
            block_config["dim"], position_encoding_size, input_size, has_class_token
        )
        self.blocks = nn.Sequential()
        self.attn_buffers = []
        local_offset_inds = None
        if max_local_offset > 0 and len(input_size) == 2:
            local_offset_inds, copy_inds_lut = get_heatmap_offset_indices(input_size, max_local_offset)
        for i in range(depth):
            block_class_i = block_class
            block_config_i = block_config.copy()
            if i in window_indices:
                if i == 0 and first_block_class is not None:
                    block_class_i = first_block_class
                elif windowed_class is not None:
                    block_class_i = windowed_class
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                if i == 0 and first_block_class is not None:
                    block_class_i = first_block_class
                elif i == (depth - 1) and last_block_class is not None:
                    block_class_i = last_block_class
                block_config_i["window_size"] = None
            block_config_i["token_drop"] = token_drop
            block_config_i["rollout_head_fusion"] = rollout_head_fusion
            block_config_i["first_frame_prune_disable"] = first_frame_prune_disable
            block_config_i["rollout_only_global"] = rollout_only_global
            block_config_i["rollout_pool_size"] = rollout_pool_size
            if i == 0 and max_local_offset > 0:
                block_config_i["heatmap_local_offset"] = True
                block_config_i["max_local_offset"] = max_local_offset
                block_config_i["local_offset_inds"] = local_offset_inds
                block_config_i["copy_inds_lut"] = copy_inds_lut
                block_config_i["offset_relative_prct_thr"] = offset_relative_prct_thr
                block_config_i["offset_using_patch_embed"] = offset_using_patch_embed

            drop_in_cur_block = 0
            if i in drop_block_indices:
                j = drop_block_indices.index(i)
                if len(num_to_drop_in_block) == 0:
                    drop_in_cur_block = round(rollout_keep_ratio * input_size[1] * input_size[0])
                else:
                    drop_in_cur_block = num_to_drop_in_block[j]
            block_config_i["num_to_drop_in_block"] = drop_in_cur_block
            block_config_i["has_class_token"] = has_class_token
            block_config_i["compute_rollout"] = drop_using_rollout or force_compute_rollout
            block_config_i["propagate_rollout"] = propagate_rollout
            block_config_i["rollout_reset_frq"] = rollout_reset_frq
            block_config_i["reset_full_process"] = reset_full_process
            block_config_i["rollout_gradual_drop"] = drop_using_rollout and len(num_to_drop_in_block) > 0
            if i == 0:
                block_config_i["first_drop_pattern"] = "" if first_drop_pattern == "odd_even" else first_drop_pattern
            if drop_using_rollout and i in rollout_block_indices:
                block_config_i["drop_using_rollout"] = drop_using_rollout
                block_config_i["rollout_keep_ratio"] = rollout_keep_ratio
                block_config_i["rollout_fg_samples_ratio"] = rollout_fg_samples_ratio
                block_config_i["rollout_bg_samples_ratio"] = rollout_bg_samples_ratio
                block_config_i["rollout_spatial_interp"] = rollout_spatial_interp
                block_config_i["rollout_gilbert_sampling"] = rollout_gilbert_sampling
                if rollout_gilbert_sampling:
                    block_config_i["gilbert_lut"] = get_gilbert_lut(input_size[1], input_size[0])

            if i == (depth - 1):
                block_config_i["block_output_buffer"] = output_buffer
                block_config_i["copy_output_dbg_mode"] = copy_output_dbg_mode
            self.blocks.append(
                getattr(blocks, block_class_i)(input_size=input_size, **block_config_i)
            )
            if len(window_indices) > 0 and (drop_using_rollout or force_compute_rollout):
                # if architecture contains windowed blocks and rollout is computed, create attention buffers for
                # reuse in rollout computation
                if i in window_indices:
                    self.attn_buffers.append(None)
                else:
                    self.attn_buffers.append(TokenBuffer())

        # token drop config
        self.token_drop = token_drop
        self.drop_block_indices = drop_block_indices
        self.num_to_drop_in_block = num_to_drop_in_block
        self.has_class_token = has_class_token
        self.late_drop_pattern = late_drop_pattern
        self.gilbert_lut = None
        self.num_patches = None
        if token_drop:
            if self.late_drop_pattern == 'gilbert':
                self.gilbert_lut = get_gilbert_lut(input_size[1], input_size[0])
        self.num_patches = input_size[1] * input_size[0] if len(input_size) == 2 else None
        self.drop_using_rollout = drop_using_rollout
        self.force_compute_rollout = force_compute_rollout
        self.rollout_final_norm = rollout_final_norm
        self.attn_wins_to_full_inds = None
        self.full_attn_size = None
        self.window_indices = window_indices
        if (drop_using_rollout or force_compute_rollout) and len(window_indices) > 0:
            if rollout_only_global:
                self.full_attn_size = (prod(input_size), prod(input_size))
            else:
                # Create indices to generate full attention matrix from the windowed attention
                d = self.blocks[window_indices[0]].window_size
                s = input_size
                p = (-s[0] % d[0], -s[1] % d[1])
                total_h = p[0] + s[0]
                total_w = p[1] + s[1]
                x_wins_num = int(total_w / d[1])
                y_wins_num = int(total_h / d[0])
                wins_inds = torch.arange(0, x_wins_num * y_wins_num)
                wins_wins_row = wins_inds // x_wins_num
                wins_wins_col = wins_inds - wins_wins_row * x_wins_num
                wins_x_offset = wins_wins_col * d[1]
                wins_y_offset = wins_wins_row * d[0]
                base_win_grid_rows, base_win_grid_cols = torch.meshgrid(torch.arange(d[0]), torch.arange(d[1]),
                                                                        indexing='ij')
                wins_grid = [base_win_grid_cols + wins_x_offset[w_ind] +
                             torch.arange(wins_y_offset[w_ind] * total_w, (wins_y_offset[w_ind] + d[0]) * total_h,
                                          total_h)[..., None].expand(-1, d[1])
                             for w_ind in range(x_wins_num * y_wins_num)]
                wins_global_inds = [torch.flatten(wins_grid[w_ind]) for w_ind in range(x_wins_num * y_wins_num)]
                wins_query_grids = [wins_global_inds[w_ind][..., None].expand(-1, d[0] * d[1]) for w_ind in
                                    range(x_wins_num * y_wins_num)]
                wins_keys_grids = [wins_global_inds[w_ind][None, ...].expand(d[0] * d[1], -1) for w_ind in
                                   range(x_wins_num * y_wins_num)]
                final_attn_w = total_w * total_h
                wins_final_inds = [
                    torch.flatten(wins_query_grids[w_ind]) * final_attn_w + torch.flatten(wins_keys_grids[w_ind]) for
                    w_ind in range(x_wins_num * y_wins_num)]
                wins_final_inds = torch.stack(wins_final_inds, dim=0)
                self.attn_wins_to_full_inds = torch.flatten(wins_final_inds)
                self.full_attn_size = (final_attn_w, final_attn_w)
        self.rollout_by_bbox = rollout_by_bbox
        self.rollout_bbox_score_thr = rollout_bbox_score_thr
        self.rollout_only_global = rollout_only_global
        assert rollout_pool_size in [None, 2, 4], 'valid values for rollout pool size are: None/2/4'
        self.rollout_pool_size = rollout_pool_size
        self.max_local_offset = max_local_offset
        self.matmul = CountedMatmul()
        self.offset_using_patch_embed = offset_using_patch_embed
        self.maskvd_prune = maskvd_prune
        self.maskvd_bbox_score_thr = maskvd_bbox_score_thr
        self.maskvd_gilbert_sampling = maskvd_gilbert_sampling
        if maskvd_gilbert_sampling:
            self.gilbert_lut = get_gilbert_lut(input_size[1], input_size[0])
        self.maskvd_add_noise = maskvd_add_noise
        if rollout_backward:
            assert rollout_pool_size is None, "rollout_backward mode is not supported with rollout pooling"
            assert rollout_only_global, "rollout_backward is supported only when rollout_only_global mode is enabled"
            assert len(self.window_indices) > 0, "rollout_backward is currently supported for vit-det arch"
        self.rollout_backward = rollout_backward

    def forward(self, x, first_frame=False, t_min1_heatmaps=None, time_step=None, t_min1_metric=None):
        patch_embed_metric = None
        if self.max_local_offset > 0 and self.offset_using_patch_embed:
            patch_embed_metric = x.clone().detach()
        x = self.position_encoding(x)
        if not (self.token_drop or self.force_compute_rollout):
            x = self.blocks(x)
        else:
            active_tokens_ind = None
            blocks_attention = []
            blocks_samples_idx = []
            tome_clusters_maps = None
            heatmaps = None
            force_keep_mask = None
            metric_t = None
            blk_in_importance = None
            if self.drop_using_rollout or self.force_compute_rollout or self.late_drop_pattern == 'gilbert':
                B, N, _ = x.size()
                n_patches = N - 1 if self.has_class_token else N
                active_tokens_ind = torch.arange(n_patches, device=x.device)[None, ...].repeat(B, 1)
                if self.drop_using_rollout or self.force_compute_rollout:
                    tome_clusters_maps = torch.arange(n_patches, device=x.device)[None, ...].repeat(B, 1)

            for i, block in enumerate(self.blocks):
                x, metric, blk_attention, active_tokens_ind, tome_clusters_maps, force_keep_mask, samples_idx,\
                t_min1_heatmaps, blk_in_importance = \
                    block(x, active_tokens_ind, tome_clusters_maps, first_frame, t_min1_heatmaps, time_step,
                          force_keep_mask, t_min1_metric, patch_embed_metric, blk_in_importance)
                blocks_attention.append(blk_attention)
                blocks_samples_idx.append(samples_idx)
                if i == 0 and self.max_local_offset > 0:
                    if self.offset_using_patch_embed:
                        metric_t = patch_embed_metric
                    else:
                        metric_t = metric.clone().detach()

            if self.drop_using_rollout or self.force_compute_rollout:
                if not self.rollout_by_bbox:
                    heatmaps = self.generate_rollout_heatmap(blocks_attention, blocks_samples_idx, B, N, x.device,
                                                             active_tokens_ind, tome_clusters_maps, first_frame,
                                                             t_min1_heatmaps, None)
                    blocks_attention = None
                    blocks_samples_idx = None
                return x, heatmaps, tome_clusters_maps, active_tokens_ind, blocks_attention, blocks_samples_idx, \
                       metric_t
        return x

    def _recombine_attn_windows_for_rollout(self, win_attn):
        attn = torch.zeros((self.full_attn_size[0] * self.full_attn_size[1],), device=win_attn.device,
                           dtype=win_attn.dtype)
        attn[self.attn_wins_to_full_inds] = torch.flatten(win_attn)
        attn = torch.reshape(attn, shape=self.full_attn_size)
        # Note that batch_size=1 is assumed here
        attn = attn[None,...]
        return attn

    def compute_rollout_heatmap_backward(self, blocks_attention, rollout_size, pruned_attn, inds_in_orig, anchors_mask,
                                         batch_size, device):

        if anchors_mask is None:
            anchors_mask = torch.ones(batch_size, rollout_size, dtype=torch.bool, device=device)
        s_l = (anchors_mask.to(torch.float32) / torch.sum(anchors_mask))[..., None]
        I = torch.eye(rollout_size, device=device).repeat(batch_size, 1, 1)

        for rev_i, blk_attention in enumerate(reversed(blocks_attention)):
            i = len(blocks_attention) - 1 - rev_i
            if i in self.window_indices:
                continue
            else:
                if not pruned_attn:
                    self.attn_buffers[i].reset_self()
                    self.attn_buffers[i](blk_attention, None)
                else:
                    blk_attention = self.attn_buffers[i](blk_attention, inds_in_orig)

            a = (blk_attention + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            s_l = self.matmul(torch.transpose(a, 1, 2), s_l)

        mask = s_l[..., 0]
        return mask

    def generate_rollout_heatmap(self, blocks_attention, blocks_samples_idx, batch_size, n_input_tokens, device,
                                 inds_in_orig, clusters_map, first_frame, t_min1_heatmaps, anchors_mask):
        # generate rollout matrix using attention matrices from all layers
        assert len(blocks_attention) == len(self.blocks), 'expecting # of attention matrices to be # of blocks'
        pruned_attn = self.full_attn_size is not None and (inds_in_orig.shape[-1] < self.full_attn_size[-1])
        rollout_size = n_input_tokens
        if self.rollout_pool_size is not None:
            rollout_size = int(n_input_tokens / self.rollout_pool_size ** 2)
            if anchors_mask is not None:
                anchors_img = anchors_mask.view(int(sqrt(n_input_tokens)), int(sqrt(n_input_tokens)))
                pooled_anchors_img = func.avg_pool2d(anchors_img[None, ...].to(torch.float32), self.rollout_pool_size)
                pooled_anchors_img = pooled_anchors_img[0] >= 0.5
                anchors_mask = torch.flatten(pooled_anchors_img)

        if self.rollout_backward:
            mask = self.compute_rollout_heatmap_backward(blocks_attention, rollout_size, pruned_attn, inds_in_orig,
                                                         anchors_mask, batch_size, device)
        else:
            rollout_mat = torch.eye(rollout_size, device=device).repeat(batch_size, 1, 1)
            I = torch.eye(rollout_size, device=device).repeat(batch_size, 1, 1)
            for i, block in enumerate(self.blocks):
                blk_attention = blocks_attention[i]
                if i in self.window_indices:
                    if self.rollout_only_global:
                        continue
                    else:
                        blk_attention = self._recombine_attn_windows_for_rollout(blk_attention)
                elif len(self.window_indices) == 0:
                    pass
                else:
                    # global attention blocks in an architecture with both global and windowed attention blocks
                    if not pruned_attn:
                        self.attn_buffers[i].reset_self()
                        self.attn_buffers[i](blk_attention, None)
                    else:
                        if self.rollout_pool_size is not None:
                            # pre-processing for pruned attention matrix with pooling
                            n_tok_orig = self.full_attn_size[-1]
                            n_tok_w = int(sqrt(n_tok_orig))
                            attn_padded = torch.zeros(blk_attention.shape[0], n_tok_orig, blk_attention.shape[-1],
                                                      dtype=blk_attention.dtype, device=blk_attention.device)
                            attn_active = torch.zeros(blk_attention.shape[0], n_tok_orig, dtype=blk_attention.dtype,
                                                      device=blk_attention.device)
                            attn_padded.scatter_(1, inds_in_orig[..., None].expand(-1, -1, blk_attention.shape[-1]),
                                                 blk_attention)
                            attn_active.scatter_(1, inds_in_orig, 1.0)
                            attn_padded = torch.swapaxes(attn_padded, 1, 2)
                            attn_padded = attn_padded.view(attn_padded.shape[0], attn_padded.shape[1], n_tok_w, n_tok_w)
                            attn_ds_dim1 = func.avg_pool2d(attn_padded, self.rollout_pool_size, divisor_override=1)
                            # calc divisor
                            attn_active = attn_active.view(attn_active.shape[0], 1, n_tok_w, n_tok_w)
                            attn_active_ds = func.avg_pool2d(attn_active, self.rollout_pool_size, divisor_override=1)
                            # save active indices, later pad using buffer reuse in remaining indices
                            active_mask = attn_active_ds > 0
                            active_mask = active_mask[:, 0].reshape(active_mask.shape[0], -1)
                            active_indices = torch.where(active_mask)[1]
                            # scale (divide by active tokens number)
                            attn_active_ds[attn_active_ds == 0] = 1.0
                            attn_ds_dim1 = attn_ds_dim1 / attn_active_ds
                            attn_ds_dim1 = attn_ds_dim1.view(attn_ds_dim1.shape[0], attn_ds_dim1.shape[1], -1)
                            blk_attention = torch.swapaxes(attn_ds_dim1, 1, 2)
                            blk_attn_active = torch.gather(blk_attention, dim=1,
                                                           index=active_indices[None, ..., None].expand(-1, -1,
                                                                                                        blk_attention.shape[
                                                                                                            -1]))
                            blk_attention = self.attn_buffers[i](blk_attn_active, active_indices[None, ...])
                        else:
                            blk_attention = self.attn_buffers[i](blk_attention, inds_in_orig)

                a = (blk_attention + 1.0 * I) / 2
                a = a / a.sum(dim=-1, keepdim=True)
                rollout_mat = self.matmul(a, rollout_mat)
                if blocks_samples_idx[i] is not None:
                    if len(self.window_indices) > 0:
                        pass
                    else:
                        rollout_mat = _reduce_rollout_mat_after_pruning(rollout_mat, blocks_samples_idx[i])
                        I = torch.eye(rollout_mat.shape[-1], device=device).repeat(batch_size, 1, 1)

            if self.has_class_token:
                mask = rollout_mat[:, 0, 1:]
            else:
                if anchors_mask is not None:
                    mask = torch.mean(rollout_mat[:, anchors_mask, :], dim=1)
                else:
                    mask = torch.mean(rollout_mat, dim=1)

        if self.rollout_pool_size is not None:
            mask_img = mask.view(mask.shape[0], int(sqrt(mask.shape[-1])), int(sqrt(mask.shape[-1])))
            mask_img_us = func.interpolate(mask_img[:, None, :], scale_factor=self.rollout_pool_size, mode='bilinear')
            mask = torch.reshape(mask_img_us[:, 0], (mask_img_us.shape[0], -1))

        if mask.shape[-1] < self.num_patches:
            full_mask = torch.zeros(mask.shape[0], self.num_patches, device=mask.device, dtype=mask.dtype)
            full_mask.scatter_(dim=1, index=inds_in_orig, src=mask)
            mask = full_mask

        # Interpolate using TOME clusters map
        if first_frame or (self.force_compute_rollout and not self.drop_using_rollout):
            heatmaps = torch.gather(mask, index=clusters_map, dim=1)
        elif self.rollout_pool_size is not None:
            heatmaps = mask
        else:
            clusters_map_wo_copy = clusters_map.clone().detach()
            clusters_map_wo_copy[clusters_map_wo_copy < 0] = 0
            heatmaps = torch.gather(mask, index=clusters_map_wo_copy, dim=1)
            heatmaps = torch.where(clusters_map == -1, t_min1_heatmaps, heatmaps)

        if self.rollout_final_norm == "sum":
            heatmaps = heatmaps / torch.sum(heatmaps, 1, keepdim=True)
        elif self.rollout_final_norm == "none":
            pass
        else:
            raise ValueError("unexpected value for 'rollout_final_norm': %s" % self.rollout_final_norm)
        return heatmaps
