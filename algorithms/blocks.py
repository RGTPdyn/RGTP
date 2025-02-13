import torch
import torch.nn as nn
import torch.nn.functional as func
from math import sqrt, prod, pow

from algorithms.base import ExtendedModule, numeric_tuple
from algorithms.counting import CountedAdd, CountedLinear, CountedMatmul
from algorithms.modules import (
    SimpleSTGTGate,
    TokenBuffer,
    TokenDeltaGate,
    TokenGate,
    MatmulDeltaAccumulator,
    MatmulBuffer,
)
from algorithms.utils import (
    DropPath,
    RelativePositionEmbedding,
    expand_row_index,
)
from utils.image import pad_to_size

LN_EPS = 1e-6


def get_first_drop_groups_inds(map_w, map_h, pattern_name):
    if pattern_name == '1in2':
        free_idx = []
        anchors_idx = []
        for row_ind in range(map_h):
            row_offset = row_ind * map_h
            row_inds = list(range(row_offset, row_offset + map_w))
            even_inds = row_inds[::2]
            odd_inds = row_inds[1::2]
            if row_ind % 2 == 0:
                anchors_idx += even_inds
                free_idx += odd_inds
            else:
                anchors_idx += odd_inds
                free_idx += even_inds
    elif pattern_name == '1in4':
        free_idx = []
        anchors_idx = []
        for row_ind in range(map_h):
            row_offset = row_ind * map_h
            row_inds = list(range(row_offset, row_offset + map_w))
            even_inds = row_inds[::2]
            odd_inds = row_inds[1::2]
            if row_ind % 2 == 0:
                anchors_idx += even_inds
                free_idx += odd_inds
            else:
                free_idx += row_inds
    elif pattern_name == '1in9':
        free_idx = []
        anchors_idx = []
        for row_ind in range(map_h):
            row_offset = row_ind * map_h
            row_inds = list(range(row_offset, row_offset + map_w))
            sample_inds = row_inds[::3]
            non_sample_inds = list(set(row_inds) - set (sample_inds))
            if row_ind % 3 == 0:
                anchors_idx += sample_inds
                free_idx += non_sample_inds
            else:
                free_idx += row_inds
        anchors_idx.sort()
        free_idx.sort()
    else:
        raise ValueError('Unexpected initial pattern name')
    return free_idx, anchors_idx


class Block(ExtendedModule):
    """
    Defines a base (non-eventful) Transformer block. Includes several
    extra features: a simple implementation of Adaptive Token
    Sampling (ATS - Fayyaz et al. 2022) and self-attention pooling.
    These features are controlled via the ats_fraction and pool_size
    parameters.
    In addition, in token_drop mode, some intermediate values from
    the self-attention calculations are passed outside and used
    in the drop decision at the backbone level.
    """

    def __init__(
        self,
        dim,
        heads,
        input_size,
        mlp_ratio,
        ats_fraction=None,
        drop_path_rate=0.0,
        relative_embedding_size=None,
        matmul_2_cast=None,
        pool_size=None,
        window_size=None,
        token_drop=False,
        num_to_drop_in_block=0,
        has_class_token=False,
        drop_using_rollout=False,
        compute_rollout=False,
        rollout_head_fusion="root_mean_squares",
        first_frame_prune_disable=False,
        rollout_keep_ratio=1.0,
        rollout_fg_samples_ratio=1.0,
        rollout_bg_samples_ratio=1.0,
        first_drop_pattern="",
        rollout_reset_frq=0,
        rollout_spatial_interp=False,
        rollout_gradual_drop=False,
        block_output_buffer=False,
        copy_output_dbg_mode=False,
        reset_full_process=False,
        rollout_only_global=True,
        rollout_pool_size=None,
        heatmap_local_offset=False,
        max_local_offset=0,
        local_offset_inds=None,
        copy_inds_lut=None,
        offset_relative_prct_thr=None,
        offset_using_patch_embed=False,
        rollout_gilbert_sampling=False,
        gilbert_lut=None,
        propagate_rollout=False,
    ):
        """
        :param dim: The number of dimensions in a token
        :param heads: The number of attention heads (None for no
        multi-headed attention)
        :param input_size: The expected size of the inputs in tokens
        :param mlp_ratio: The ratio of the MLP dimensionality to the
        token dimensionality
        :param ats_fraction: The fraction of tokens to retain if
        using Adaptive Token Sampling (ATS)
        :param drop_path_rate: Drop path ratio (for use when training)
        :param relative_embedding_size: The size (in tokens) assumed for
        relative position embeddings
        :param matmul_2_cast: Typecast for the attention-value product
        (None, "float16", or "bfloat16"). Helps save some memory when
        using an A-gate, without a noticeable impact on accuracy.
        :param pool_size: Pooling ratio to use with self-attention
        pooling.
        :param window_size: Self-attention window size (None to use
        global, non-windowed attention).
        :param token_drop: indicates that token drop mode is enabled
        :param num_to_drop_in_block: number of tokens to drop in block
        :param has_class_token: Whether to add an extra class token
        :param drop_using_rollout: Use attention rollout in token
        drop decision
        :param compute_rollout: compute current frame rollout. this is
        used if any of the blocks uses rollout for dropping or for
        visualization mode
        :param rollout_head_fusion: method to combine attention from
         all heads during rollout computation (mean/root_mean_squares)
        :param first_frame_prune_disable: disable the spatial based
        pruning on the first frame
        :param rollout_keep_ratio: the ratio of tokens to keep using
        the rollout inorfmation ("foreground tokens")
        :param rollout_fg_samples_ratio: the ratio of tokens to sample
        from the "foreground region"
        :param rollout_bg_samples_ratio: the ratio of tokens to sample
        from the "background region"
        :param first_drop_pattern: tokens groups pattern to use in the
         first pruning stage (default is "" meaning uniform sampling
         or other criteria is used for sampling, e.g. rollout maps)
        :param rollout_reset_frq: if set as integer value k > 0, every
        k frames the rollout map from previous frame will not be used
        and pruning will be done using current frame information only
        :param rollout_spatial_interp: if set to true, the rollout map
        for pruned tokens (in a frame where pruning is based on rollout)
        will be interpolated using "semantic" distance in current frame.
        Note that in the "reset frames" this is always the case. if set
        to false, rollout is interpolated from the previous frame at the
        same position.
        :param rollout_gradual_drop: indicates that pruning is done gradually
        across layers. At each step, the tokens with the lowest rollout score
        are pruned.
        :param block_output_buffer: maintain a buffer of the blcok's output
         (in full resolution)
        :param reset_full_process: perform full processing at "rollout
        reset frames"
        :param rollout_only_global: compute attention rollout based only
        on the transformer blocks with global attention. This is an
        approximated computation for architectures with "windowed"
        attention blcoks
        :param rollout_pool_size: pooling size to apply to the attention
        matrices for computation saving (supported values: 2,4,None).
        By default set to None (no pooling)
        :param heatmap_local_offset: when set to True, the rollout
        heatmap from t-1 will be locally modified by matching tokens
        in the current frame. Typically set to True at Layer 0.
        :param max_local_offset: if larger than 0, the rollout value
        at frame t+1 will be taken from the most similar token at a
        (2*max_local_offset+1) * (2*max_local_offset+1) neighborhood
        in frame t
        :param local_offset_inds: if heatmap_local_offset mode is
        enabled, this contains a tensor of indices for token distance
        calculation between frames (input_h,input_w,n_offsets)
        :param copy_inds_lut: tensor of size input_h*input_w*n_offsets
        containing the samples indices from the orig corridnates and
        each combination of x, y, offset_ind
        :param offset_relative_prct_thr: positive value indicating
        the minimal percentage increase compared to "keep in same place"
        needed to allow applying offset for the heatmap at the given token
        :param offset_using_patch_embed: compute the distance for offset
        calculation on the patch embedding before first layer
        :param rollout_gilbert_sampling: sample tokens from foreground
        and background regions defined by rollout using gilbert
        :param gilbert_lut: maps between indices in the input sapce
        to indices in gilbert curve
        :param propagate_rollout: do gradual drop per layer using
        input rollout map propagation in current frame's layers
        """
        super().__init__()
        self.heads = heads
        self.input_size = tuple(input_size)
        if ats_fraction is not None:
            assert pool_size is None
            assert window_size is None
            assert not (ats_fraction < 0.0 or ats_fraction > 1.0)
        assert not (drop_path_rate < 0.0 or drop_path_rate > 1.0)
        assert matmul_2_cast in [None, "float16", "bfloat16"]
        self.ats_fraction = ats_fraction
        self.last_ats_indices = None
        self.matmul_2_cast = matmul_2_cast
        if pool_size is None:
            self.pool_size = None
        else:
            self.pool_size = numeric_tuple(pool_size, length=2)
        if window_size is None:
            self.window_size = None
            attention_size = input_size
        else:
            self.window_size = numeric_tuple(window_size, length=2)
            attention_size = self.window_size
            if relative_embedding_size is not None:
                relative_embedding_size = self.window_size
        self.scale = sqrt(dim // heads)

        # Set up submodules.
        self.input_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.qkv = CountedLinear(in_features=dim, out_features=dim * 3)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        if relative_embedding_size is not None:
            self.relative_position = RelativePositionEmbedding(
                attention_size,
                relative_embedding_size,
                dim // heads,
                pool_size=self.pool_size,
            )
        else:
            self.relative_position = None
        self.matmul = CountedMatmul()
        self.projection = CountedLinear(in_features=dim, out_features=dim)
        self.add = CountedAdd()
        self.mlp_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp_1 = CountedLinear(in_features=dim, out_features=dim * mlp_ratio)
        self.gelu = nn.GELU()
        self.mlp_2 = CountedLinear(in_features=dim * mlp_ratio, out_features=dim)
        # token drop config
        self.token_drop = token_drop
        self.num_to_drop_in_block = num_to_drop_in_block
        self.has_class_token = has_class_token
        self.drop_using_rollout = drop_using_rollout
        self.compute_rollout = compute_rollout
        self.rollout_head_fusion = rollout_head_fusion
        self.first_frame_prune_disable = first_frame_prune_disable
        self.rollout_keep_ratio = rollout_keep_ratio
        self.rollout_fg_samples_ratio = rollout_fg_samples_ratio
        self.rollout_bg_samples_ratio = rollout_bg_samples_ratio
        self.first_drop_a_idx = None
        self.first_drop_b_idx = None
        if len(first_drop_pattern) > 0:
            self.first_drop_a_idx, self.first_drop_b_idx = \
                get_first_drop_groups_inds(input_size[1], input_size[0], first_drop_pattern)
        self.rollout_reset_frq = rollout_reset_frq
        self.rollout_spatial_interp = rollout_spatial_interp
        self.rollout_gradual_drop = rollout_gradual_drop
        self.output_buffer = None
        self.copy_output_dbg_mode = copy_output_dbg_mode
        if block_output_buffer:
            self.output_buffer = TokenBuffer()
        self.reset_full_process = reset_full_process
        self.rollout_only_global = rollout_only_global
        self.rollout_pool_size = rollout_pool_size
        self.heatmap_local_offset = heatmap_local_offset
        self.max_local_offset = max_local_offset
        self.local_offset_inds = local_offset_inds
        self.copy_inds_lut = copy_inds_lut
        self.offset_relative_prct_thr = offset_relative_prct_thr
        self.offset_using_patch_embed = offset_using_patch_embed
        self.rollout_gilbert_sampling = rollout_gilbert_sampling
        self.gilbert_lut = gilbert_lut
        self.propagate_rollout = propagate_rollout

    def forward(self, x, active_tokens_ind=None, tome_clusters_maps=None, first_frame=False, t_min1_heatmaps=None,
                time_step=None, force_keep_mask=None, t_min1_metric=None, patch_embed_metric=None,
                blk_in_importance=None):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Linearly project x into qkv space.
        x = self.qkv(x)
        metric = None
        if self.token_drop or self.compute_rollout:
            _, k, _ = self._partition_heads(x)
            metric = k.mean(1)

        # Compute attention on the qkv representation.
        x, ats_indices, blk_attn_mat, prop_attn_mat = self._forward_attention((x, active_tokens_ind))
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)

        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Apply token dropping
        samples_idx = None
        blk_out_importance = None
        is_reset_frame = self.rollout_reset_frq > 0 and time_step % self.rollout_reset_frq == 0
        reset_full_processing = self.reset_full_process and self.token_drop and not first_frame and is_reset_frame
        if not first_frame and not is_reset_frame and self.heatmap_local_offset:
            assert prod(self.input_size) == t_min1_heatmaps.shape[-1], 'expected offset to be calculated on full' \
                                                                       ' feature maps'
            t_metric = patch_embed_metric if self.offset_using_patch_embed else metric
            t_min1_heatmaps = self.apply_offset_to_heatmap(t_min1_heatmaps, t_metric, t_min1_metric)

        score_for_drop = None
        if not(first_frame or is_reset_frame) and self.compute_rollout and self.propagate_rollout:
            if self.has_class_token:
                prop_attn_mat = prop_attn_mat[:, 1:, 1:]
            if blk_in_importance is None:
                blk_in_importance = t_min1_heatmaps[..., None]
            blk_prop_importance = self.matmul(torch.inverse(prop_attn_mat), blk_in_importance)
            # normalize to sum to 1
            summed_importance = torch.sum(blk_prop_importance, 1, keepdim=True)
            blk_prop_importance = blk_prop_importance / summed_importance
            score_for_drop = blk_prop_importance[..., 0]
        if self.token_drop and self.num_to_drop_in_block > 0 and not(first_frame and self.first_frame_prune_disable)\
                and not reset_full_processing:
            if not (self.compute_rollout and self.propagate_rollout):
                t_min1_heatmap_for_drop = None if t_min1_heatmaps is None else t_min1_heatmaps.clone()
                if self.drop_using_rollout and not first_frame:
                    # sample rollout maps for layer ind > 0
                    if active_tokens_ind.shape[1] < t_min1_heatmaps.shape[1]:
                        t_min1_heatmap_for_drop = torch.gather(t_min1_heatmap_for_drop, dim=1, index=active_tokens_ind)
                score_for_drop = t_min1_heatmap_for_drop
            x, samples_idx, pruned_idx, pruned_match_idx, force_keep_mask = \
                self.drop_tokens(x, metric, self.num_to_drop_in_block, first_frame, score_for_drop, time_step,
                                 force_keep_mask)
            active_tokens_ind, tome_clusters_maps = \
                self._update_active_tokens_and_clusters(active_tokens_ind, tome_clusters_maps, samples_idx, pruned_idx,
                                                        pruned_match_idx)
        if self.compute_rollout and self.propagate_rollout and not(first_frame or is_reset_frame):
            if samples_idx is not None:
                patches_samples_idx = samples_idx
                if self.has_class_token:
                    patches_samples_idx = samples_idx[:, 1:]
                    patches_samples_idx = patches_samples_idx - 1
                blk_out_importance = torch.gather(blk_prop_importance, dim=1, index=patches_samples_idx[..., None])
            else:
                blk_out_importance = blk_prop_importance

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        if self.output_buffer is not None:
            if reset_full_processing:
                self.output_buffer.reset_self()
            if self.copy_output_dbg_mode and not reset_full_processing and not first_frame:
                x = self.output_buffer.b
            else:
                x = self.output_buffer(x, active_tokens_ind)
        if not (self.token_drop or self.compute_rollout):
            return x
        else:
            return x, metric, blk_attn_mat, active_tokens_ind, tome_clusters_maps, force_keep_mask, samples_idx,\
                   t_min1_heatmaps, blk_out_importance

    def reset_self(self):
        self.last_ats_indices = None

    # A simple version of the method from
    # "Adaptive Token Sampling for Efficient Vision Transformers"
    # (Fayyaz et al., ECCV 2022)
    # For now we just use the top-k version of ATS (select the tokens
    # with the k highest scores). Using CDF-based token sampling should
    # also be possible, but it would be more complex to implement (we
    # would need a mechanism for masking the K' < K active tokens in
    # gates and buffers).
    def _adaptive_token_sampling(self, a, v):
        if self.ats_fraction is None:
            return a, None

        class_scores = a[..., 0]
        raw_scores = class_scores * torch.linalg.vector_norm(v[...], dim=-1)
        scores = raw_scores / raw_scores[..., 1:].sum(dim=-1, keepdim=True)

        # Always select the class token.
        scores[..., 0] = float("inf")

        # Sum scores over heads.
        scores = scores.sum(dim=-3)

        # Add +1 for the class token
        n_select = int(self.ats_fraction * (scores.shape[-1] - 1)) + 1

        # Select the k tokens with the highest scores.
        ats_indices = scores.topk(n_select, sorted=False)[1]

        # Sort the token indices (for stabilization). This seems to
        # work pretty well, although we could probably come up with
        # better/more sophisticated. E.g., we could try to find the
        # permutation of indices that minimized some norm between the
        # previous and current ats_indices.
        ats_indices = self._stabilize_ats_indices(ats_indices)
        self.last_ats_indices = ats_indices

        return (
            a.gather(dim=-2, index=expand_row_index(ats_indices, a.shape)),
            ats_indices,
        )

    def find_closest_kept_token(self, kept_idx, pruned_idx, metric):
        m_kept = torch.gather(metric, dim=1, index=kept_idx[..., None].expand(-1, -1, metric.shape[2]))
        m_pruned = torch.gather(metric, dim=1, index=pruned_idx[..., None].expand(-1, -1, metric.shape[2]))
        dist_mat = self.matmul(m_pruned, m_kept.transpose(-1, -2))
        node_max, node_idx = dist_mat.max(dim=-1)
        pruned_match_idx = torch.gather(kept_idx, dim=1, index=node_idx)
        return pruned_match_idx

    def apply_offset_to_heatmap(self, in_heatmap, t_metric, t_min1_metric):
        local_offset_inds = self.local_offset_inds.to(in_heatmap.device)  # (in_h,in_w,2*max_local_offset+1)
        copy_inds_lut = self.copy_inds_lut.to(in_heatmap.device)
        if self.has_class_token:
            t_metric = t_metric[:, 1:]  # (b, in_h*in_w, c)
            t_min1_metric = t_min1_metric[:, 1:]  # (b, in_h*in_w, c)
        b = t_min1_metric.shape[0]
        c = t_min1_metric.shape[-1]
        # pad t_min1_metric
        t_min1_metric = t_min1_metric.reshape(b, self.input_size[0], self.input_size[1], c)  # (b, in_h, in_w, c)
        k = self.max_local_offset
        t_min1_metric = func.pad(t_min1_metric, (0, 0, k, k, k, k), mode='constant', value=0) # (b, in_h+2k, in_w+2k, c)
        # -> # (b, (in_h+2k)*(in_w+2k), c)
        t_min1_metric = t_min1_metric.reshape(b, -1, c)
        # for each token in t_metric tensor, find (2k+1)**2 indices to compare to in t_min1_metric tensor
        # -> (in_h * in_w * (2k+1)**2)
        local_offset_inds = torch.flatten(local_offset_inds.reshape(-1, local_offset_inds.shape[-1]))
        # -> (b, in_h * in_w * (2k+1)**2, c)
        t_min1_metric_refs = torch.gather(t_min1_metric, 1, local_offset_inds[None, ..., None].expand(b, -1, c))
        # create 3d tensors to allow bmm
        # t_metric: [b, in_h*in_w, c] -> [b*in_h*in_w, 1, c]
        t_metric = t_metric.reshape(-1, 1, c)
        # t_min1_metric_refs: [b, in_h*in_w*(2k+1)**2, c] -> [b*in_h*in_w, (2k+1)**2, c]
        t_min1_metric_refs = t_min1_metric_refs.reshape(b, self.input_size[0] * self.input_size[1], (2 * k + 1) ** 2, c)
        t_min1_metric_refs = t_min1_metric_refs.reshape(-1, (2 * k + 1) ** 2, c)
        # bmm
        sim_scores = self.matmul(t_metric, torch.transpose(t_min1_metric_refs, 1, 2))
        if self.offset_relative_prct_thr is not None:
            no_offset_ind = (2 * k + 1) ** 2 // 2
            no_offset_score = sim_scores[:, :, no_offset_ind][..., None]
            no_offset_score[no_offset_score <= 0] = 0.1
            rel_sim_scores = sim_scores / no_offset_score
            max_rel_sim = rel_sim_scores.max(2, keepdims=True)[0]
            invalid_for_offset = torch.squeeze(max_rel_sim < (100 + self.offset_relative_prct_thr) / 100)
            sim_scores[invalid_for_offset, :, no_offset_ind] = 100

        sim_scores[sim_scores == 0.0] = -100
        match_offset_inds = torch.argmax(sim_scores[:, 0, :], dim=1)
        match_offset_inds = match_offset_inds.reshape(b, -1)
        # apply the offset to the input heatmap
        n_offsets = (2 * self.max_local_offset + 1) ** 2
        tok_global_start = torch.arange(match_offset_inds.shape[-1], device=match_offset_inds.device) * n_offsets
        match_inds_in_lut = tok_global_start[None, :].expand(b, -1) + match_offset_inds
        lut_expanded = copy_inds_lut[None, ...].expand(b, -1)
        match_copy_inds = torch.gather(lut_expanded, 1, match_inds_in_lut)
        out_heatmap = torch.gather(in_heatmap, 1, match_copy_inds)
        return out_heatmap

    def calc_pruned_match_idx_for_rollout_drop(self, samples_idx, pruned_idx, m, device):
        if self.rollout_spatial_interp:
            pruned_match_idx = self.find_closest_kept_token(samples_idx, pruned_idx, m)
        else:
            pruned_match_idx = -1 * torch.ones(pruned_idx.shape[0], pruned_idx.shape[1], dtype=torch.int64,
                                               device=device)
        return pruned_match_idx

    def rollout_based_gradual_drop(self, x_patches, force_keep_mask, t_min1_heatmaps, num_to_drop, assume_sorted=False):
        if assume_sorted and not self.propagate_rollout:
            all_idx = torch.arange(x_patches.shape[-2], device=x_patches.device)
            all_idx = all_idx[None, ...].expand(x_patches.shape[0], -1)
            samples_idx = all_idx[:, :-num_to_drop]
            pruned_idx = all_idx[:, -num_to_drop:]
        else:
            keep_scores = t_min1_heatmaps.clone()
            if not self.propagate_rollout:
                keep_scores[force_keep_mask] = 10.0
            sort_idx = keep_scores.argsort(dim=-1, descending=True)
            samples_idx = sort_idx[:, :-num_to_drop]
            pruned_idx = sort_idx[:, -num_to_drop:]
        n, t1, c = x_patches.shape
        x_0 = x_patches.gather(dim=-2, index=samples_idx[..., None].expand(-1, -1, c))
        if not self.propagate_rollout:
            force_keep_mask = force_keep_mask.gather(dim=-1, index=samples_idx)
        return x_0, samples_idx, pruned_idx, force_keep_mask

    def naive_spatial_drop(self, x_patches, metric):
        if self.first_drop_a_idx is not None and self.first_drop_b_idx is not None:
            samples_idx = torch.tensor(self.first_drop_b_idx, device=x_patches.device)
        else:
            tok_keep = x_patches.shape[1] - self.num_to_drop_in_block
            samples_idx = torch.round(torch.linspace(0, x_patches.shape[1] - 1, tok_keep, device=x_patches.device)
                                      ).type(torch.int64)
            if self.compute_rollout:
                all_samples_idx = torch.arange(x_patches.shape[1], device=x_patches.device)
                pruned_idx = torch.squeeze(all_samples_idx[torch.nonzero(torch.isin(all_samples_idx, samples_idx)
                                                                         == 0)])
        x_0 = x_patches[:, samples_idx, :]
        samples_idx = samples_idx[None, ...].repeat(x_0.shape[0], 1)
        if self.compute_rollout:
            if self.first_drop_a_idx is None:
                pass
            else:
                pruned_idx = torch.tensor(self.first_drop_a_idx, device=x_patches.device)
            pruned_idx = pruned_idx[None, ...].repeat(x_0.shape[0], 1)
            # Find the closest kept token for each pruned token
            pruned_match_idx = self.find_closest_kept_token(samples_idx, pruned_idx, metric)
        else:
            pruned_idx = None
            pruned_match_idx = None
        return x_0, samples_idx, pruned_idx, pruned_match_idx

    def drop_tokens(self, x, metric, r, first_frame, t_min1_heatmaps, time_step, force_keep_mask):
        x_patches = x
        m = metric
        if self.has_class_token:
            x_cls, x_patches = torch.split(x, [1, x.shape[1] - 1], dim=1)
            _, m = torch.split(metric, [1, metric.shape[1] - 1], dim=1)

        if self.drop_using_rollout and not first_frame:
            if self.rollout_reset_frq > 0 and time_step > 0 and time_step % self.rollout_reset_frq == 0:
                # SPATIAL BASED PRUNING (RESET ROLLOUT)
                x_0, samples_idx, pruned_idx, pruned_match_idx = self.naive_spatial_drop(x_patches, m)
            else:
                # TEMPORAL BASED PRUNING (USE ROLLOUT)
                # Check if this is the first drop in the frame
                orig_num_tokens = self.input_size[0] * self.input_size[1]
                cur_num_tokens = x_patches.shape[1]
                is_first_drop = orig_num_tokens == cur_num_tokens
                if self.rollout_gradual_drop and not is_first_drop:
                    x_0, samples_idx, pruned_idx, force_keep_mask = \
                        self.rollout_based_gradual_drop(x_patches, force_keep_mask, t_min1_heatmaps, r,
                                                        assume_sorted=True)
                    pruned_match_idx = self.calc_pruned_match_idx_for_rollout_drop(samples_idx, pruned_idx, m,
                                                                                   x.device)
                else:
                    # find foreground tokens masks
                    keep_scores = t_min1_heatmaps.clone()
                    keep_sort_inds = keep_scores.argsort(dim=-1, descending=True)
                    num_to_keep = round(self.rollout_keep_ratio * x_patches.shape[1])
                    fg_inds = keep_sort_inds[:, :num_to_keep][..., None]
                    if self.rollout_fg_samples_ratio < 1.0:
                        fg_sample_stride = round(1 / self.rollout_fg_samples_ratio)
                        fg_samples_all_slice_idx = torch.arange(fg_inds.shape[1])
                        fg_keep_idx = fg_samples_all_slice_idx[::fg_sample_stride]
                        fg_prune_idx = torch.squeeze(
                            fg_samples_all_slice_idx[torch.nonzero(torch.isin(fg_samples_all_slice_idx, fg_keep_idx) == 0)])
                        if self.rollout_gilbert_sampling:
                            gilbert_luts = torch.tensor(self.gilbert_lut, device=fg_inds.device)[None, ...].repeat(fg_inds.shape[0], 1)[..., None]
                            fg_tokens_curve_inds = torch.gather(gilbert_luts, 1, fg_inds)
                            fg_tokens_sort_inds = torch.argsort(fg_tokens_curve_inds, dim=1)
                            fg_keep_idx = fg_tokens_sort_inds[:, fg_keep_idx]
                            fg_prune_idx = fg_tokens_sort_inds[:, fg_prune_idx]
                            fg_keep_inds = torch.gather(fg_inds, 1, fg_keep_idx)
                        else:
                            fg_keep_inds = fg_inds[:, fg_keep_idx]
                    else:
                        fg_keep_inds = fg_inds
                    bg_inds = torch.sort(keep_sort_inds[:, num_to_keep:], dim=-1)[0]
                    if self.rollout_bg_samples_ratio > 0.0:
                        bg_sample_stride = round(1 / self.rollout_bg_samples_ratio)
                        bg_samples_all_slice_idx = torch.arange(bg_inds.shape[1])
                        bg_keep_idx = bg_samples_all_slice_idx[::bg_sample_stride]
                        bg_prune_idx = torch.squeeze(bg_samples_all_slice_idx[torch.nonzero(torch.isin(bg_samples_all_slice_idx, bg_keep_idx) == 0)])
                        if self.rollout_gilbert_sampling:
                            gilbert_luts = torch.tensor(self.gilbert_lut, device=bg_inds.device)[None, ...].repeat(bg_inds.shape[0], 1)
                            bg_tokens_curve_inds = torch.gather(gilbert_luts, 1, bg_inds)
                            bg_tokens_sort_inds = torch.argsort(bg_tokens_curve_inds, dim=1)
                            bg_keep_idx = bg_tokens_sort_inds[:, bg_keep_idx]
                            bg_prune_idx = bg_tokens_sort_inds[:, bg_prune_idx]
                            bg_keep_inds = torch.gather(bg_inds, 1, bg_keep_idx)[..., None]
                        else:
                            bg_keep_inds = bg_inds[:, bg_keep_idx][..., None]
                    else:
                        bg_keep_inds = None
                        bg_prune_idx = torch.arange(bg_inds.shape[1], device=bg_inds.device)
                    if self.rollout_gradual_drop:
                        # first drop, so create the original "force keep mask" and then prune according to sorted
                        # rollout values
                        force_keep_mask = None
                        if not self.propagate_rollout:
                            force_keep_mask = torch.zeros(x_patches.shape[0], x_patches.shape[1], dtype=torch.bool,
                                                          device=x.device)
                            force_keep_mask.scatter_(dim=1, index=fg_keep_inds[:, :, 0], value=True)
                            force_keep_mask.scatter_(dim=1, index=bg_keep_inds[:, :, 0], value=True)
                        x_0, samples_idx, pruned_idx, force_keep_mask = \
                            self.rollout_based_gradual_drop(x_patches, force_keep_mask, t_min1_heatmaps, r,
                                                            assume_sorted=False)
                    else:
                        # "One time" rollout drop (all pruning is done in 1 layer) / rollout propagate mode at first layer
                        fg_x = x_patches.gather(dim=-2, index=fg_keep_inds.expand(-1, -1, x_patches.shape[-1]))
                        if bg_keep_inds is not None:
                            bg_x = x_patches.gather(dim=-2, index=bg_keep_inds.expand(-1, -1, x_patches.shape[-1]))
                            x_0 = torch.cat([fg_x, bg_x], dim=1)
                            samples_idx = torch.cat([fg_keep_inds, bg_keep_inds], dim=1)[..., 0]
                        else:
                            x_0 = fg_x
                            samples_idx = fg_keep_inds[..., 0]
                        if self.rollout_fg_samples_ratio < 1.0:
                            if self.rollout_gilbert_sampling:
                                pruned_idx = torch.gather(bg_inds, 1, bg_prune_idx)
                                fg_pruned_idx = torch.gather(fg_inds, 1, fg_prune_idx)[..., 0]
                            else:
                                pruned_idx = bg_inds[:, bg_prune_idx]
                                fg_pruned_idx = fg_inds[:, fg_prune_idx, 0]
                            pruned_idx = torch.cat([fg_pruned_idx, pruned_idx], dim=1)
                        else:
                            if self.rollout_gilbert_sampling and self.rollout_bg_samples_ratio > 0:
                                pruned_idx = torch.gather(bg_inds, 1, bg_prune_idx)
                            else:
                                pruned_idx = bg_inds[:, bg_prune_idx]
                    pruned_match_idx = self.calc_pruned_match_idx_for_rollout_drop(samples_idx, pruned_idx, m,
                                                                                   x.device)

        if self.has_class_token:
            x_0 = torch.cat([x_cls, x_0], dim=1)
            samples_idx = samples_idx + 1
            cls_sample = torch.zeros(samples_idx.shape[0], 1, dtype=samples_idx.dtype, device=samples_idx.device)
            samples_idx = torch.cat([cls_sample, samples_idx], dim=1)

        return x_0, samples_idx, pruned_idx, pruned_match_idx, force_keep_mask

    def _summarize_attention_for_rollout(self, attention):
        if not self.compute_rollout or (self.window_size is not None and self.rollout_only_global):
            return None, None
        if self.rollout_head_fusion == "mean":
            fused_attn = attention.mean(axis=1)
        elif self.rollout_head_fusion == "root_mean_squares":
            attn_sumsq = torch.sum(torch.square(attention), dim=1)
            fused_attn = torch.sqrt(attn_sumsq / attention.shape[1])
        else:
            raise ValueError("Attention head fusion type not supported")
        prop_attn_mat = None
        if self.propagate_rollout:
            prop_attn_mat = fused_attn.clone()
        pruned_qs_and_ks = fused_attn.shape[-2] < prod(self.input_size) and fused_attn.shape[-1] < prod(self.input_size)
        # pooling attention matrices
        # In case of pruning both the queries AND the keys and values in global attention blocks, don't pool here
        if self.rollout_pool_size is not None and not(self.window_size is None and pruned_qs_and_ks):
            sym_atten_mat = fused_attn.shape[-2] == fused_attn.shape[-1]
            full_attn_imgs = fused_attn.view(fused_attn.shape[0], fused_attn.shape[1], *self.input_size)
            pooled_attn_imgs = func.avg_pool2d(full_attn_imgs, self.rollout_pool_size, divisor_override=1)
            pooled_attn_dim2 = pooled_attn_imgs.view(fused_attn.shape[0], fused_attn.shape[1],
                                                     pooled_attn_imgs.shape[-2] * pooled_attn_imgs.shape[-1])
            if sym_atten_mat:
                # original attention matrix is symmetric, can pool spatially twice
                attn_dim1_pooled_imgs = pooled_attn_dim2.view(fused_attn.shape[0], *self.input_size,
                                                              pooled_attn_dim2.shape[-1])
                attn_dim1_pooled_imgs = torch.swapaxes(attn_dim1_pooled_imgs, 2, 3)
                attn_dim1_pooled_imgs = torch.swapaxes(attn_dim1_pooled_imgs, 1, 2)
                pooled_attn_2dims = func.avg_pool2d(attn_dim1_pooled_imgs, self.rollout_pool_size)
                pooled_attn_2dims = pooled_attn_2dims.view(pooled_attn_2dims.shape[0], pooled_attn_2dims.shape[1],
                                                         pooled_attn_2dims.shape[-2] * pooled_attn_2dims.shape[-1])
                fused_attn = torch.swapaxes(pooled_attn_2dims, 1, 2)
            else:
                fused_attn = pooled_attn_dim2
        return fused_attn, prop_attn_mat

    def _update_active_tokens_and_clusters(self, active_tokens_ind, tome_clusters_maps, samples_idx, pruned_idx,
                                           pruned_match_idx):
        if active_tokens_ind is None:
            return active_tokens_ind, tome_clusters_maps
        prev_active_tokens_ind = active_tokens_ind.clone().detach()
        patches_samples_idx = samples_idx
        if self.has_class_token:
            patches_samples_idx = samples_idx[:, 1:]
            patches_samples_idx = patches_samples_idx - 1
        active_tokens_ind = torch.gather(active_tokens_ind, dim=1, index=patches_samples_idx)
        # update TOME clusters maps
        # WITH ROLLOUT PRUNING
        if not self.rollout_spatial_interp:
            for clip_ind in range(tome_clusters_maps.shape[0]):
                pruned_by_rollout_mask = pruned_match_idx[clip_ind] == -1
                pruned_by_tome_mask = pruned_match_idx[clip_ind] >= 0
                if torch.any(pruned_by_tome_mask):
                    pruned_idx_in_img = prev_active_tokens_ind[clip_ind, pruned_idx[clip_ind, pruned_by_tome_mask]]
                    pruned_match_idx_in_img = prev_active_tokens_ind[clip_ind, pruned_match_idx[clip_ind, pruned_by_tome_mask]]
                    tome_clusters_maps[clip_ind, pruned_idx_in_img] = pruned_match_idx_in_img
                if torch.any(pruned_by_rollout_mask):
                    pruned_idx_in_img = prev_active_tokens_ind[clip_ind, pruned_idx[clip_ind, pruned_by_rollout_mask]]
                    tome_clusters_maps[clip_ind, pruned_idx_in_img] = -1
                # if any tokens pointed to the pruned tokens, make sure they now point to the new cluster
                default_indices = torch.arange(tome_clusters_maps.shape[1], device=tome_clusters_maps.device)
                fixed_indices = torch.where(tome_clusters_maps[clip_ind] == -1, default_indices,
                                            tome_clusters_maps[clip_ind])
                tome_clusters_maps[clip_ind] = tome_clusters_maps[clip_ind, fixed_indices]
        else:
            # WITHOUT ROLLOUT PRUNING / ROLLOUT SPATIAL INTERPOLATION:
            pruned_idx_in_img = torch.gather(prev_active_tokens_ind, dim=1, index=pruned_idx)
            pruned_match_idx_in_img = torch.gather(prev_active_tokens_ind, dim=1, index=pruned_match_idx)
            tome_clusters_maps.scatter_(dim=1, index=pruned_idx_in_img, src=pruned_match_idx_in_img)
            # if any tokens pointed to the pruned tokens, make sure they now point to the new cluster
            tome_clusters_maps = torch.gather(tome_clusters_maps, dim=1, index=tome_clusters_maps)

        return active_tokens_ind, tome_clusters_maps

    def _cast_matmul_2(self, x, v):
        old_dtype = x.dtype
        if self.matmul_2_cast is not None:
            dtype = getattr(torch, self.matmul_2_cast)
            x = x.to(dtype)
            v = v.to(dtype)
        return x, v, old_dtype

    def _compute_window_padding(self):
        pad_h = -self.input_size[0] % self.window_size[0]
        pad_w = -self.input_size[1] % self.window_size[1]
        return pad_h, pad_w

    @staticmethod
    def _gather_ats_skip(skip_1, ats_indices):
        if ats_indices is None:
            return skip_1
        else:
            return skip_1.gather(
                dim=-2, index=expand_row_index(ats_indices, skip_1.shape)
            )

    def _forward_attention(self, x):
        # (batch, token, dim)
        x, active_inds = (x[0], x[1]) if type(x) is tuple else (x, None)

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        # For non-windowed attention, sample only the active queries
        rel_pos_padding = False
        pad_first_dim_only = False
        if self.window_size is None and active_inds is not None:
            if hasattr(self, 'qkv_accumulator'):
                # reuse in global attention block case
                if active_inds.shape[-1] < q.shape[-2]:
                    q = torch.gather(q, dim=2, index=active_inds[None, ..., None].expand(q.shape[0], q.shape[1], -1,
                                                                                         q.shape[3]))
                    rel_pos_padding = True
                    pad_first_dim_only = True
            elif active_inds.shape[-1] < prod(self.input_size):
                # prune in global attention block case
                rel_pos_padding = True
                pad_first_dim_only = False

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            if rel_pos_padding:
                x_padded = torch.zeros(x.shape[0], x.shape[1], prod(self.input_size), x.shape[3], dtype=x.dtype,
                                       device=x.device)
                x_padded.scatter_(2, active_inds[None, ..., None].expand(x.shape[0], x.shape[1], -1, x.shape[3]), x)
                x = x_padded
                q_padded = torch.zeros(q.shape[0], q.shape[1], prod(self.input_size), q.shape[3], dtype=q.dtype,
                                       device=q.device)
                q_padded.scatter_(2, active_inds[None, ..., None].expand(q.shape[0], q.shape[1], -1, q.shape[3]), q)
                q = q_padded
                if not pad_first_dim_only:
                    x_padded = torch.zeros(x.shape[0], x.shape[1], prod(self.input_size), prod(self.input_size),
                                           dtype=x.dtype, device=x.device)
                    x_padded.scatter_(3, active_inds[None, None, ...].expand(x.shape[0], x.shape[1], x.shape[2], -1), x)
                    x = x_padded

            x = self.relative_position(x, q)
            if rel_pos_padding:
                x = torch.gather(x, 2, active_inds[None, ..., None].expand(x.shape[0], x.shape[1], -1, x.shape[3]))
                if not pad_first_dim_only:
                    x = torch.gather(x, 3, active_inds[None, None, ...].expand(x.shape[0], x.shape[1], x.shape[2], -1))

        x = x.softmax(dim=-1)

        # Adaptive token sampling is a noop if self.ats_fraction is None.
        x, ats_indices = self._adaptive_token_sampling(x, v)

        # block attention is None if self.compute_rollout is False
        block_attention, prop_attn_mat = self._summarize_attention_for_rollout(x)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        return x, ats_indices, block_attention, prop_attn_mat

    def _forward_mlp(self, x):
        x = self.mlp_1(x)
        x = self.gelu(x)
        x = self.mlp_2(x)
        return x

    def _partition_heads(self, x):
        # (batch, token, dim)

        x = x.view(x.shape[:-1] + (3, self.heads, x.shape[-1] // (3 * self.heads)))
        q, k, v = x.permute(2, 0, 3, 1, 4)
        # (batch, heads, token, dim / heads)

        return q, k, v

    def _partition_windows(self, x, in_qkv_domain):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        # (batch, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(x.shape[:1] + self.input_size + x.shape[2:])
        # (batch, height, width, dim)

        if any(p):
            s = x.shape
            pad_tensor = torch.zeros(
                (1,) * (x.ndim - 1) + s[-1:], dtype=x.dtype, device=x.device
            )

            # The attention computation expects padded tokens to equal
            # _forward_qkv(zero). If x has already been mapped to the
            # QKV domain, then we need to transform the padded zero
            # values to the QKV domain. Only the bias portion of the
            # linear transform has an effect on the zero padding vector.
            if in_qkv_domain:
                pad_tensor = self.qkv.forward_bias(pad_tensor)

            # Pad to a multiple of the window size.
            # func.pad seems broken (see the comments in pad_to_size).
            # In the meantime we'll use pad_to_size.
            # x = func.pad(x, (0, 0, 0, p[1], 0, p[0]))
            x = pad_to_size(x, (s[-3] + p[0], s[-2] + p[1], s[-1]), pad_tensor)
            # (batch, height, width, dim)

        # Partition into windows.
        s = x.shape
        x = x.view(-1, s[-3] // d[0], d[0], s[-2] // d[1], d[1], s[-1])
        x = x.transpose(-3, -4)
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Re-flatten the spatial dimensions. Can't use x.view here
        # because of the transpose.
        x = x.reshape(-1, prod(d), s[-1])
        # (batch * window, token, dim)

        return x

    def _pool_tokens(self, x):
        # (batch, heads, token, dim)

        if self.pool_size is None:
            return x
        w = self.input_size if (self.window_size is None) else self.window_size
        s = x.shape

        # Can't use x.view here because of the permutation in
        # _partition_heads.
        x = x.reshape((-1,) + w + x.shape[-1:])
        # (batch * heads, token_y, token_x, dim)

        x = x.permute(0, 3, 1, 2)
        x = func.avg_pool2d(x, self.pool_size)
        # (batch * heads, dim, token_y, token_x)

        x = x.permute(0, 2, 3, 1)
        # (batch * heads, token_y, token_x, dim)

        x = x.view(s[:-2] + (-1,) + s[-1:])
        # (batch, heads, token, dim)

        return x

    @staticmethod
    def _recombine_heads(x):
        # (batch, heads, token, dim / heads)

        # Can't use x.view here because of the permutation.
        x = x.permute(0, 2, 1, 3)
        x_reshaped = x.reshape(x.shape[:-2] + (-1,))
        # (batch, token, dim)

        # We assume that x.reshape actually copies the data. We can run
        # into problems if this is not the case, i.e., we may end up
        # with a gate being passed a raw reference to an accumulator
        # state. For an example, see EventfulMatmul1Block.
        assert x.data_ptr() != x_reshaped.data_ptr()
        x = x_reshaped

        return x

    def _recombine_windows(self, x):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        s = self.input_size
        total_h = p[0] + s[0]
        total_w = p[1] + s[1]
        # (batch * window, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(-1, total_h // d[0], total_w // d[1], d[0], d[1], x.shape[-1])
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Recombine the window partitions. Can't use x.view here because
        # of the transpose.
        x = x.transpose(-3, -4)
        x = x.reshape(-1, total_h, total_w, x.shape[-1])
        # (batch, height, width, dim)

        # Remove padding.
        if any(p):
            x = x[:, : s[0], : s[1]]
            # (batch, height, width, dim)

        # Re-flatten the spatial dimensions.
        x = x.flatten(start_dim=1, end_dim=2)
        # (batch, token, dim)

        return x

    def _stabilize_ats_indices(self, ats_indices):
        ats_indices = ats_indices.sort(dim=-1)[0]
        if self.last_ats_indices is None:
            return ats_indices

        # Faster on the CPU
        new_indices = ats_indices.flatten(end_dim=-2).cpu()
        old_indices = self.last_ats_indices.flatten(end_dim=-2).cpu()
        stabilized = old_indices.clone()
        for i in range(new_indices.shape[0]):
            old_not_in_new = torch.isin(old_indices[i], new_indices[i], invert=True)
            new_not_in_old = torch.isin(new_indices[i], old_indices[i], invert=True)
            stabilized[i, old_not_in_new] = new_indices[i, new_not_in_old]
        return stabilized.to(ats_indices.device).view(ats_indices.shape)

    def _uncast_matmul_2(self, x, old_dtype):
        if self.matmul_2_cast is not None:
            x = x.to(old_dtype)
        return x


class EventfulTokenwiseBlock(Block):
    """
    A Transformer block that adds eventfulness to token-wise operations.
    """

    def __init__(self, gate_before_ln=False, stgt=False, **super_kwargs):
        """
        :param gate_before_ln: Determines whether token gates are placed
        before or after layer norm operations
        :param stgt: Whether to use the SimpleSTGTGate (instead of our
        TokenGate) for benchmarking
        :param super_kwargs: Kwargs for the super class (Block)
        """
        super().__init__(**super_kwargs)
        self.gate_before_ln = gate_before_ln
        token_gate_class = SimpleSTGTGate if stgt else TokenGate
        self.qkv_gate = token_gate_class()
        self.qkv_accumulator = TokenBuffer()
        self.projection_gate = token_gate_class()
        self.projection_accumulator = TokenBuffer()
        self.mlp_gate = token_gate_class()
        self.mlp_accumulator = TokenBuffer()

    def forward(self, x):
        skip_1, x, index = self._forward_pre_attention(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices, _, _ = self._forward_attention(x)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention(x, skip_1)
        return x

    def _forward_post_attention(self, x, skip_1):
        # Gate-accumulator block 2
        x, index = self.projection_gate(x)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        x = self.add(self.drop_path(x), skip_1)
        skip_2 = x

        # Gate-accumulator block 3
        if self.gate_before_ln:
            x, index = self.mlp_gate(x)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x)
        x = self._forward_mlp(x)
        x = self.mlp_accumulator(x, index)
        x = self.add(self.drop_path(x), skip_2)

        return x

    def _forward_pre_attention(self, x):
        skip_1 = x

        # Gate-accumulator block 1
        if self.gate_before_ln:
            x, index = self.qkv_gate(x)
            x = self.input_layer_norm(x)
        else:
            x = self.input_layer_norm(x)
            x, index = self.qkv_gate(x)
        x = self.qkv(x)
        return skip_1, x, index


class EventfulMatmul1Block(EventfulTokenwiseBlock):
    """
    An EventfulTokenWiseBlock that adds eventfulness to the query-key
    product (in addition to token-wise operations).
    """

    def __init__(self, **super_kwargs):
        """
        :param super_kwargs: Kwargs for the super class (
        EventfulTokenwiseBlock)
        """
        super().__init__(**super_kwargs)

        # self._pool_index assumes that the input size is divisible by
        # the pooling size.
        if self.pool_size is not None:
            assert all(s % p == 0 for s, p in zip(self.input_size, self.pool_size))

        # This class only supports non-windowed attention for now.
        assert self.window_size is None

        self.matmul_accumulator_1 = MatmulBuffer()

    def forward(self, x):
        skip_1, x, index = self._forward_pre_attention(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices = self._forward_attention((x, index))
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention(x, skip_1)
        return x

    def _forward_attention(self, x):
        x, v, _ = self._forward_matmul_1(x)
        x, ats_indices = self._adaptive_token_sampling(x, v)
        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        x = self._recombine_heads(x)
        x = self._uncast_matmul_2(x, old_dtype)
        return x, ats_indices

    def _forward_matmul_1(self, x):
        x, index = x
        q, k, v = self._partition_heads(x)
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)
        index_k = self._pool_index(index)

        # See comment in Block._forward_attention.
        x = self.matmul_accumulator_1(
            q / self.scale, k.transpose(-2, -1), index, index_k
        )

        if self.relative_position is not None:
            # We need inplace=False because x is a direct reference to
            # an accumulator state.
            x = self.relative_position(x, q, inplace=False)
        x = x.softmax(dim=-1)
        return x, v, index_k

    def _pool_index(self, index):
        if (self.pool_size is None) or (index is None):
            return index
        width = self.input_size[1]
        index_y = index.div(width, rounding_mode="floor")
        index_x = index.remainder(width)
        index_y = index_y.div(self.pool_size[0], rounding_mode="floor")
        index_x = index_x.div(self.pool_size[1], rounding_mode="floor")
        index = index_y * (width // self.pool_size[1]) + index_x

        # Calling .unique() still works if there are multiple items in
        # the batch. However, the output size along dim=-1 will be the
        # largest of the individual output sizes. This could result in
        # some redundant downstream computation.
        index = index.unique(dim=-1)
        return index


class EventfulBlock(EventfulMatmul1Block):
    """
    An EventfulMatmul1Block that also adds eventfulness to the
    attention-value product.
    """
    def __init__(self, **super_kwargs):
        """
        :param super_kwargs: Kwargs for the super class (
        EventfulTokenwiseBlock)
        """
        super().__init__(**super_kwargs)
        self.v_gate = TokenDeltaGate()
        self.matmul_gate = TokenDeltaGate(structure="col")
        self.matmul_accumulator_2 = MatmulDeltaAccumulator()

    def _forward_attention(self, a):
        a, v, index_k = self._forward_matmul_1(a)

        a, v, old_dtype = self._cast_matmul_2(a, v)
        a, ats_indices = self._adaptive_token_sampling(a, v)
        if not self.matmul_2_cast:
            # We clone v here because it may be a direct reference to
            # self.qkv_accumulator.a.
            v = v.clone()
        v_n_tilde, v_delta_tilde, index_v = self.v_gate(v, forced_index=index_k)
        a_n_tilde, a_delta_tilde, _ = self.matmul_gate(a, forced_index=index_v)
        a = self.matmul_accumulator_2(
            a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde
        )

        a = self._recombine_heads(a)
        a = self._uncast_matmul_2(a, old_dtype)
        return a, ats_indices


class TokenwisePrunedBlock(Block):
    """
    A Transformer block with pruning applied to token-wise operations. Missing tokens in attention computation are
    reused from previous frame buffer.
    """

    def __init__(self, do_a=False, do_b=False, **super_kwargs):
        """
        :param gate_before_ln: Determines whether token gates are placed
        before or after layer norm operations
        :param stgt: Whether to use the SimpleSTGTGate (instead of our
        TokenGate) for benchmarking
        :param super_kwargs: Kwargs for the super class (Block)
        """
        super().__init__(**super_kwargs)
        self.qkv_accumulator = TokenBuffer()

    def forward(self, x, active_tokens_ind=None, tome_clusters_maps=None, first_frame=False, t_min1_heatmaps=None,
                time_step=None, force_keep_mask=None, t_min1_metric=None, patch_embed_metric=None,
                blk_in_importance=None):
        skip_1, x, metric = self._forward_pre_attention(x)
        reset_full_processing = self.reset_full_process and self.token_drop and not first_frame and\
                                self.rollout_reset_frq > 0 and time_step % self.rollout_reset_frq == 0
        if reset_full_processing:
            self.qkv_accumulator.reset_self()
        x = self.qkv_accumulator(x, active_tokens_ind)

        # Compute attention on the qkv representation.
        x, ats_indices, blk_attn_mat, _ = self._forward_attention((x, active_tokens_ind))
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        if (not first_frame) and (self.window_size is not None):
            # after attention, keep only the active tokens
            x = torch.gather(x, dim=1, index=active_tokens_ind[..., None].expand(-1, -1, x.shape[-1]))

        x, samples_idx = self._forward_post_attention(x, skip_1, first_frame, reset_full_processing, active_tokens_ind)

        if not (self.token_drop or self.compute_rollout):
            return x
        else:
            return x, metric, blk_attn_mat, active_tokens_ind, tome_clusters_maps, force_keep_mask, samples_idx,\
                   t_min1_heatmaps, None

    def _forward_post_attention(self, x, skip_1, first_frame, reset_full_processing, active_tokens_ind):
        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # For Vit-Det Assuming pruning only takes place on the first block which doesn't use attention cache
        samples_idx = None

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)

        if self.output_buffer is not None:
            if reset_full_processing:
                self.output_buffer.reset_self()
            if self.copy_output_dbg_mode and not reset_full_processing and not first_frame:
                x = self.output_buffer.b
            else:
                x = self.output_buffer(x, active_tokens_ind)

        return x, samples_idx

    def _forward_pre_attention(self, x):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Linearly project x into qkv space.
        x = self.qkv(x)
        metric = None
        if self.token_drop:
            _, k, _ = self._partition_heads(x)
            metric = k.mean(1)

        return skip_1, x, metric
