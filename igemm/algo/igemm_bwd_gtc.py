################################################################################
# 
#  MIT License
# 
#  Copyright (c) 2020 Advanced Micro Devices, Inc.
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
# 
################################################################################
# pylint: disable=maybe-no-member
from ..codegen import *
from .fma_main_loop import *
from .igemm_base import *
from .global_memory import *
from .shared_memory import *
from .utility import *
from .thread_mapping import *
from .coalescing_store import *

IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1 = 0
IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C1_C0 = 1
IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B = 4
IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 = 5


def _find_non_1_index_in_list(list_object):
    result_list = list()
    for idx, item in enumerate(list_object):
        assert type(item) is int
        if item != 1:
            result_list.append(idx)
    return result_list

class macro_igemm_bwd_gtc_out_update_os_t(mc_base_t):
    def __init__(self, mc, data_byte):
        mc_base_t.__init__(self, mc)
        self.data_byte = data_byte
    def name(self):
        return '.v_bwd_gtc_out_update_os'
    def __call__(self, v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_out_os, v_out_os_base, v_out_iho, v_out_iwo, s_wo, v_tmp'.format(self.name())):
            self._emit(f"; from ho, wo, os_base, compute final offset")
            self._emit(f"v_mad_u32_u24 v[\\v_tmp], s[\\s_wo], v[\\v_out_iho], v[\\v_out_iwo]")
            self._emit(f"v_lshl_add_u32 v[\\v_out_os], v[\\v_tmp], {igemm_log2(self.data_byte)}, v[\\v_out_os_base]")

class macro_igemm_bwd_gtc_out_update_hw_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def name(self):
        return '.v_bwd_gtc_out_update_hw'
    def __call__(self, v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_out_iho, v_out_iwo, v_out_dslice_ih, v_out_dslice_iw, v_out_dslice_iy, v_out_dslice_ix, s_dtile_dy_neg, s_dtile_dx_neg'.format(self.name())):
            self._emit(f"; dslice_y,dslice_h -> oh, dslice_x,dslice_w -> ow")
            self._emit(f"v_mad_i32_i24 v[\\v_out_iho], s[\\s_dtile_dy_neg], v[\\v_out_dslice_iy], v[\\v_out_dslice_ih]")
            self._emit(f"v_mad_i32_i24 v[\\v_out_iwo], s[\\s_dtile_dx_neg], v[\\v_out_dslice_ix], v[\\v_out_dslice_iw]")

class macro_igemm_bwd_gtc_wei_update_os_t(mc_base_t):
    def __init__(self, mc, data_byte):
        mc_base_t.__init__(self, mc)
        self.data_byte = data_byte
    def name(self):
        return '.v_bwd_gtc_wei_update_os'
    def __call__(self, v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp):
        return '{} {}, {}, {}, {}, {}, {}'.format(self.name(), v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_wei_os, v_wei_os_base, v_wei_iy, v_wei_ix, s_x, v_tmp'.format(self.name())):
            self._emit(f"; from y, x, os_base, compute final offset")
            self._emit(f"v_mad_u32_u24 v[\\v_tmp], v[\\v_wei_iy], s[\\s_x], v[\\v_wei_ix]")
            self._emit(f"v_lshl_add_u32 v[\\v_wei_os], v[\\v_tmp], {igemm_log2(self.data_byte)}, v[\\v_wei_os_base]")


class macro_igemm_bwd_gtc_wei_update_yx_t(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def name(self):
        return '.v_bwd_gtc_wei_update_yx'
    def __call__(self, v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_wei_iy, v_wei_ix, v_wei_dslice_iy, v_wei_dslice_ix, s_dtile_y, s_dtile_x, v_dtile_iy, v_dtile_ix'.format(self.name())):
            self._emit(f"v_mad_u32_u24 v[\\v_wei_iy], s[\\s_dtile_y], v[\\v_wei_dslice_iy], v[\\v_dtile_iy]")
            self._emit(f"v_mad_u32_u24 v[\\v_wei_ix], s[\\s_dtile_x], v[\\v_wei_dslice_ix], v[\\v_dtile_ix]")


class macro_igemm_bwd_gtc_set_flag_hw(mc_base_t):
    def __init__(self, mc):
        mc_base_t.__init__(self, mc)
    def name(self):
        return '.v_set_flag_hw'
    def __call__(self, v_flag, v_ih, v_iw, s_h, s_w):
        return '{} {}, {}, {}, {}, {}'.format(self.name(), v_flag, v_ih, v_iw, s_h, s_w)
    def emit(self):
        with self._emit_macro_indented('.macro {} v_flag, v_ih, v_iw, s_h, s_w'.format(self.name())):
            self._emit(f"v_cmp_gt_u32 vcc, s[\\s_h], v[\\v_ih]")
            self._emit(f"v_cndmask_b32 v[\\v_flag], 0, 1, vcc")
            self._emit(f"v_cmp_gt_u32 vcc, s[\\s_w], v[\\v_iw]")
            self._emit(f"v_cndmask_b32 v[\\v_flag], 0, v[\\v_flag], vcc")


class macro_igemm_bwd_gtc_move_slice_window_k_dsy_dsx(mc_base_t):
    '''
    optimized move slice approach. 
    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable

    def name(self):
        return '.s_bwd_gtc_move_slice_window_k_dsy_dsx'

    def __call__(self, v_move_slice_k_ik1, v_move_slice_k_idsy, v_move_slice_k_idsx, s_gemm_k_num_k1, s_gemm_k_num_dsy, s_gemm_k_num_dsx, s_move_slice_k_k1, s_move_slice_k_dsy, s_move_slice_k_dsx, v_out_os_base, v_wei_os_base, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_move_slice_k_ik1, v_move_slice_k_idsy, v_move_slice_k_idsx, s_gemm_k_num_k1, s_gemm_k_num_dsy, s_gemm_k_num_dsx, s_move_slice_k_k1, s_move_slice_k_dsy, s_move_slice_k_dsx, v_out_os_base, v_wei_os_base, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff)
    def init_stride_k(self, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff, s_move_slice_k_k1):
        '''
        s_out_stride_k, s_wei_stride_k, s_move_slice_k_k1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4

        c_k0, c_k1e, c_c0, c_c1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_n0, c_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_k0, t_k1e, t_c0, t_c1  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_n0, t_n1b = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_k0, n_k1e = c_k0 * t_k0, c_k1e * t_k1e
        unmerge_sub_k = self.tunable.unmerge_sub_k
        assert unmerge_sub_k % n_k0 == 0
        unmerge_sub_k1 = unmerge_sub_k // n_k0
        assert n_k1e % unmerge_sub_k1 == 0

        diff_k0_k1 = self.tunable.gemm_k_per_block - unmerge_sub_k1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_out_stride_k_k0_k1_diff}], {diff_k0_k1}, s[{s_out_stride_k}]")
            self._emit(f"s_mul_i32 s[{s_wei_stride_k_k0_k1_diff}], {diff_k0_k1}, s[{s_wei_stride_k}]")
            self._emit(f"s_mul_i32 s[{s_out_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_out_stride_k}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s_wei_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_wei_stride_k}]  ; might be 0 or larger")

        return self._get_deferred()

    def emit(self):
        # unmerge_sub_k1 = self.unmerge_sub_k1
        '''
        This is indeed a multi-dimension add-carry operation
        e.g. if want to compute a 3d (merged) dimension index [iz, iy, ix], with dimension length of [nz, ny, nx] in each.
        suppose we want to add a specific value this merged dimension.
        1) if want to add 1, it is simple.
            ix += 1
            if ix >= nx:
                ix = 0
                iy += 1     # carry to iy
            if iy >= ny:
                iy = 0
                iz += 1     # carry to iz
            if iz >= nz:
                pass        # the final dimension indeed can be ignored
        
        2) if we want to add N
            # first, find out how many steps in each dimension needed to add
            stride_x = N % nx               # -> usually can store in sgpr
            stride_y = (N//nx) % ny         # -> usually can store in sgpr
            stride_z = (N//(nx*ny)) % nz    # -> usually can store in sgpr

            # then do the add-carry
            ix += stride_x
            if ix >= nx:
                ix -= nx    # ! note here, no longer set 0
                iy += 1     # carry to iy
            iy += stride_y
            if iy >= ny:
                iy -= ny    # ! note here, no longer set 0
                iz += 1     # carry to iz
            iz += stride_z
            if iz >= nz:
                pass        # the final dimension indeed can be ignored
        '''
        with self._emit_macro_indented('.macro {} v_move_slice_k_ik1, v_move_slice_k_idsy, v_move_slice_k_idsx, s_gemm_k_num_k1, s_gemm_k_num_dsy, s_gemm_k_num_dsx, s_move_slice_k_k1, s_move_slice_k_dsy, s_move_slice_k_dsx, v_out_os_base, v_wei_os_base, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff'.format(self.name())):
            # k0, k1e is unmerge.  k1e is merged from k1, e
            self._emit(f"v_add_u32 v[\\v_move_slice_k_idsx], s[\\s_move_slice_k_dsx], v[\\v_move_slice_k_idsx]")
            self._emit(f"v_cmpx_le_u32 vcc, s[\\s_gemm_k_num_dsx], v[\\v_move_slice_k_idsx]")
            #self._emit(f"v_mov_b32 v[\\v_move_slice_k_idsx], 0")
            self._emit(f"v_subrev_u32 v[\\v_move_slice_k_idsx], s[\\s_gemm_k_num_dsx], v[\\v_move_slice_k_idsx]")
            self._emit(f"v_add_u32 v[\\v_move_slice_k_idsy], 1, v[\\v_move_slice_k_idsy]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()
            self._emit(f"v_add_u32 v[\\v_move_slice_k_idsy], s[\\s_move_slice_k_dsy], v[\\v_move_slice_k_idsy]")
            self._emit(f"v_cmpx_le_u32 vcc, s[\\s_gemm_k_num_dsy], v[\\v_move_slice_k_idsy]")
            #self._emit(f"v_mov_b32 v[\\v_move_slice_k_idsy], 0")
            self._emit(f"v_subrev_u32 v[\\v_move_slice_k_idsy], s[\\s_gemm_k_num_dsy], v[\\v_move_slice_k_idsy]")
            self._emit(f"v_add_u32 v[\\v_move_slice_k_ik1], 1, v[\\v_move_slice_k_ik1]")
            self._emit(f"v_add_u32 v[\\v_out_os_base], s[\\s_out_stride_k], v[\\v_out_os_base]")
            self._emit(f"v_add_u32 v[\\v_wei_os_base], s[\\s_wei_stride_k], v[\\v_wei_os_base]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()
            self._emit(f"v_add_u32 v[\\v_move_slice_k_ik1], s[\\s_move_slice_k_k1], v[\\v_move_slice_k_ik1]")
            self._emit(f"v_add_u32 v[\\v_out_os_base], s[\\s_out_stride_k_k1], v[\\v_out_os_base]")
            self._emit(f"v_add_u32 v[\\v_wei_os_base], s[\\s_wei_stride_k_k1], v[\\v_wei_os_base]")
            self._emit(f"v_cmpx_le_u32 vcc, s[\\s_gemm_k_num_k1], v[\\v_move_slice_k_ik1]")
            #self._emit(f"v_mov_b32 v[\\v_move_slice_k_ik1], 0")
            self._emit(f"v_subrev_u32 v[\\v_move_slice_k_ik1], s[\\s_gemm_k_num_k1], v[\\v_move_slice_k_ik1]")
            self._emit(f"v_add_u32 v[\\v_out_os_base], s[\\s_out_stride_k_k0_k1_diff], v[\\v_out_os_base]")
            self._emit(f"v_add_u32 v[\\v_wei_os_base], s[\\s_wei_stride_k_k0_k1_diff], v[\\v_wei_os_base]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()

class macro_igemm_bwd_gtc_move_slice_window_k(mc_base_t):
    '''
    optimized move slice approach.
    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        assert self.tunable.nxe == 0, "this is for nxe 0 only"

    def name(self):
        return '.s_bwd_gtc_move_slice_window_k'

    def __call__(self, v_move_slice_k_ik1, s_gemm_k_num_k1, s_move_slice_k_k1, v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff):
        return '{} {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.name(),
            v_move_slice_k_ik1, s_gemm_k_num_k1, s_move_slice_k_k1, v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff)
    def init_stride_k(self, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff, s_move_slice_k_k1):
        '''
        s_out_stride_k, s_wei_stride_k, s_move_slice_k_k1 is known value, want to compute other
        '''
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4

        c_k0, c_k1e, c_c0, c_c1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_n0, c_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_k0, t_k1e, t_c0, t_c1  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_n0, t_n1b = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_k0, n_k1e = c_k0 * t_k0, c_k1e * t_k1e
        unmerge_sub_k = self.tunable.unmerge_sub_k
        assert unmerge_sub_k % n_k0 == 0
        unmerge_sub_k1 = unmerge_sub_k // n_k0
        assert n_k1e % unmerge_sub_k1 == 0

        diff_k0_k1 = self.tunable.gemm_k_per_block - unmerge_sub_k1 # !!! the diff of 2 unmerged dimension (like K=K0*K1)

        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_out_stride_k_k0_k1_diff}], {diff_k0_k1}, s[{s_out_stride_k}]")
            self._emit(f"s_mul_i32 s[{s_wei_stride_k_k0_k1_diff}], {diff_k0_k1}, s[{s_wei_stride_k}]")
            self._emit(f"s_mul_i32 s[{s_out_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_out_stride_k}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s_wei_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_wei_stride_k}]  ; might be 0 or larger")

        return self._get_deferred()

    def emit(self):
        with self._emit_macro_indented('.macro {} v_move_slice_k_ik1, s_gemm_k_num_k1, s_move_slice_k_k1, v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1, s_out_stride_k_k0_k1_diff, s_wei_stride_k_k0_k1_diff'.format(self.name())):
            self._emit(f"v_add_u32 v[\\v_move_slice_k_ik1], s[\\s_move_slice_k_k1], v[\\v_move_slice_k_ik1]")
            self._emit(f"v_add_u32 v[\\v_out_os], s[\\s_out_stride_k_k1], v[\\v_out_os]")
            self._emit(f"v_add_u32 v[\\v_wei_os], s[\\s_wei_stride_k_k1], v[\\v_wei_os]")
            self._emit(f"v_cmpx_le_u32 vcc, s[\\s_gemm_k_num_k1], v[\\v_move_slice_k_ik1]")
            self._emit(f"v_subrev_u32 v[\\v_move_slice_k_ik1], s[\\s_gemm_k_num_k1], v[\\v_move_slice_k_ik1]")
            self._emit(f"v_add_u32 v[\\v_out_os], s[\\s_out_stride_k_k0_k1_diff], v[\\v_out_os]")
            self._emit(f"v_add_u32 v[\\v_wei_os], s[\\s_wei_stride_k_k0_k1_diff], v[\\v_wei_os]")
            self._emit(f"s_mov_b64 exec, -1")
            self._emit_empty_line()

class macro_igemm_bwd_gtc_move_slice_window_k_1d(mc_base_t):
    '''
    optimized move slice approach. 
    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        assert self.tunable.nxe == 0, "this is for nxe 0 only"

        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4
        assert len(t_ta) == 4 and len(t_tb) == 4

        c_k0, c_k1e, c_c0, c_c1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_n0, c_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        t_k0, t_k1e, t_c0, t_c1  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_n0, t_n1b = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        n_k0, n_k1e = c_k0 * t_k0, c_k1e * t_k1e

        # assert (n_k0 == 1 and n_k1e != 1) or (n_k0 != 1 and n_k1e == 1)
        assert (n_k0 == 1 and n_k1e != 1)  # indeed in this case will assume only k1 direction non-1. only k0 non-1 is meaningless

    def name(self):
        return '.s_bwd_gtc_move_slice_window_k_1d'

    def __call__(self, v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1):
        return '{} {}, {}, {}, {}'.format(self.name(),
            v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1)
    def init_stride_k(self, s_out_stride_k, s_wei_stride_k, s_out_stride_k_k1, s_wei_stride_k_k1, s_move_slice_k_k1):
        '''
        s_out_stride_k, s_wei_stride_k, s_move_slice_k_k1 is known value, want to compute other
        '''
        with self._deferred_context():
            self._emit(f"s_mul_i32 s[{s_out_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_out_stride_k}]  ; might be 0 or larger")
            self._emit(f"s_mul_i32 s[{s_wei_stride_k_k1}], s[{s_move_slice_k_k1}], s[{s_wei_stride_k}]  ; might be 0 or larger")
        return self._get_deferred()

    def emit(self):
        with self._emit_macro_indented('.macro {} v_out_os, v_wei_os, s_out_stride_k_k1, s_wei_stride_k_k1'.format(self.name())):
            self._emit(f"v_add_u32 v[\\v_out_os], s[\\s_out_stride_k_k1], v[\\v_out_os]")
            self._emit(f"v_add_u32 v[\\v_wei_os], s[\\s_wei_stride_k_k1], v[\\v_wei_os]")
            self._emit_empty_line()


class igemm_bwd_gtc_t(mc_base_t):
    '''
    k -> k0, k1
    c -> c0, c1
    n -> n0, n1
    ho, wo -> b
    dslice_y, dslice_x -> e

    gemm_m -> c0*c1
    gemm_k -> k0*k1e
    gemm_n -> n0*n1b

    tensor a: k0*k1e*c0*c1
    tensor b: k0*k1e*n0*n1b

              thread_lengths            cluster_lengths
    tensor a: t_k0*t_k1e*t_c0*t_c1      c_k0*c_k1e*c_c0*c_c1
    tensor b: t_k0*t_k1e*t_n0*t_n1b     c_k0*c_k1e*c_n0*c_n1b

                      tensor a                      tensor b
    thread_lengths  : t_k0, t_k1e, t_c0, t_c1   t_k0, t_k1e, t_n0, t_n1b
    cluster_lengths : c_k0, c_k1e, c_c0, c_c1   c_k0, c_k1e, c_n0, c_n1b

    for the k1e, n1b, thread_lengths no longer check per thread stride in k1*e or n1*b
    but cluster lengths will check.

    '''
    def __init__(self, mc, tunable):
        assert type(tunable) is igemm_gtc_tunable_parameter_t
        mc_base_t.__init__(self, mc)
        self.tunable = tunable
        self.global_load_out = self.global_load_out_t(mc, self)
        self.global_load_wei = self.global_load_wei_t(mc, self)
        self.shared_store_out = self.shared_store_out_t(mc, self)
        self.shared_store_wei = self.shared_store_wei_t(mc, self)

        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        self.out_thread_copy_ndim = len(out_thread_copy_index)
        self.wei_thread_copy_ndim = len(wei_thread_copy_index)
        assert self.out_thread_copy_ndim in (1, 2)
        assert self.wei_thread_copy_ndim in (1, 2)

        ctrl_thread_mapping = ctrl_thread_mapping_t()
                #                        ->      MR x  NR x ML1 x NL1 x ML0 x NL0
        ctrl_thread_mapping.thread_lengths = [self.tunable.gemm_m_repeat, self.tunable.gemm_n_repeat, 1, 1, self.tunable.gemm_m_per_thread, self.tunable.gemm_n_per_thread]
        ctrl_thread_mapping.cluster_lengths = [1, 1, self.tunable.gemm_m_level1_cluster, self.tunable.gemm_n_level1_cluster, self.tunable.gemm_m_level0_cluster, self.tunable.gemm_n_level0_cluster]
        self.thread_mapping = igemm_thread_mapping_t(self.mc, ctrl_thread_mapping)


        self.coalescing_store_groups = igemm_next_pow2(self.tunable.coalescing_store_groups)
        ctrl_coalescing_store = ctrl_coalescing_store_t()
        ctrl_coalescing_store.ctm = ctrl_thread_mapping
        ctrl_coalescing_store.coalescing_groups = self.coalescing_store_groups
        ctrl_coalescing_store.data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        ctrl_coalescing_store.vector_write_out = 1                      # TODO: some cases this can be set to other value
        ctrl_coalescing_store.block_size = self.tunable.block_size

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
        ctrl_coalescing_store.gemm_m_m0_m1 = [n_c0, n_c1]
        if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C1_C0:
            ctrl_coalescing_store.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M1_M0


        ctrl_coalescing_store.adjust_optimal_coalescing_groups()        # in m1_m0 order, must adjust 
        self.coalescing_store = igemm_coalescing_store_t(mc, ctrl_coalescing_store)

        '''
         in generic tensor contraction, gemm_m direction always is *good* dimension, fwd:k0*k1, bwd:c0*c1, wrw:k0*k1
         hence we always want to split coalescing groups along m direction, to store c matrix
        '''
        assert (self.tunable.gemm_m_per_thread * self.tunable.gemm_m_repeat) % self.coalescing_store_groups == 0, \
            f"coalescing store groups should be divided by thread m {self.tunable.gemm_m_per_thread}x{self.tunable.gemm_m_repeat}"

        self.label_out = f"L_{self.name()}_out"
        self.dict_shifted_stride = dict()


        self.karg = self.kernel_karg_t(mc, self)
        self.sgpr = self.kernel_sgpr_t(mc, self)
        self.vgpr = self.kernel_vgpr_t(mc, self)


    def name(self):
        return igemm_gtc_encode_kernel_name(self.tunable)

    def try_shift_stride(self, gpr, shifter):
        assert type(gpr) is sym_t
        with self._deferred_context():
            if gpr.label not in self.dict_shifted_stride:
                self.dict_shifted_stride[gpr.label] = gpr
                self._emit(f"s_lshl_b32 s[{gpr()}], s[{gpr()}], {shifter}")
        return self._get_deferred()

    def is_1d_move_slice_k(self):
        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
        if self.tunable.nxe != 0:
            return False        # if not nxe 0, it is possible that we can do move slice, but that will lead to extra index calculation
        if n_k1e != 1 and n_k0 == 1:
            return True
        # it is meanless to let n_k1e==1 and n_k0!=1
        return False

    def get_lds_gemm_m_gemm_n_order(self):
        def need_reverse_order(x0, x1):
            if x0 != 1 and x1 == 1:
                return True
            if x0 > x1:
                return True
            return False

        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()

        gemm_n_order = IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(t_n0, t_n1b):
                gemm_n_order = IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0

        gemm_m_order = IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1
        if self.tunable.allow_lds_reorder:
            if need_reverse_order(t_c0, t_c1):
                gemm_m_order = IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C1_C0

        return gemm_m_order, gemm_n_order

    class global_load_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            return m_out_2d_global_load.get_issues()

        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load output")
                if self.outer.tunable.nxe != 0:
                    self._emit(f".v_clear_nc {v.v_gld_b()}, {m_out_2d_global_load.ctrl.length_d0 * m_out_2d_global_load.ctrl.length_d1}")
                    self._emit(f"v_cmp_eq_u32 vcc, 1, v[{v.v_out_flag()}]")
                    self._emit(f"s_and_saveexec_b64 s[{s.s_tmp(4)}:{s.s_tmp(5)}], vcc")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_out_2d_global_load(v.v_gld_b(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_out_offset()))
                else:
                    self._emit(m_out_2d_global_load(v.v_gld_b(), s.s_p_out(), v.v_out_os(), s_out_stride_d0(), s_out_stride_d1(), s.s_tmp()))
                if self.outer.tunable.nxe != 0:
                    self._emit(f"s_or_b64 exec, exec, s[{s.s_tmp(4)}:{s.s_tmp(5)}]")
            return self._get_deferred()

    class global_load_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            return m_wei_2d_global_load.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr

            m_out_2d_global_load, m_wei_2d_global_load = self.outer.get_macro_global_load()
            s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.outer.get_symbol_global_load_s_stride_d0_d1()
            with self._deferred_context():
                self._emit(f"; load weight")
                if self.outer.tunable.precache_soffset:
                    self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset()))
                else:
                    self._emit(m_wei_2d_global_load(v.v_gld_a(), s.s_p_wei(), v.v_wei_os(), s_wei_stride_d0(), s_wei_stride_d1(), s.s_tmp()))
            return self._get_deferred() 

    class shared_store_out_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return  m_out_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_out_2d_shared_store(v.v_gld_b(), v.v_sst_b_os()))
            return self._get_deferred()

    class shared_store_wei_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer
        def get_issues(self):
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            return m_wei_2d_shared_store.get_issues()
        
        def __call__(self):
            s = self.outer.sgpr
            v = self.outer.vgpr
            m_out_2d_shared_store, m_wei_2d_shared_store = self.outer.get_macro_shared_store()
            with self._deferred_context():
                self._emit(m_wei_2d_shared_store(v.v_gld_a(), v.v_sst_a_os()))
            return self._get_deferred()

    class kernel_karg_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer         = outer

            self.k_p_in          = sym_t("k_p_in",          0)
            self.k_p_wei         = sym_t("k_p_wei",         8)
            self.k_p_out         = sym_t("k_p_out",         16)
            self.k_hi            = sym_t("k_hi",            24)
            self.k_wi            = sym_t("k_wi",            28)
            self.k_n             = sym_t("k_n",             32)
            self.k_k             = sym_t("k_k",             36)
            self.k_c             = sym_t("k_c",             40)
            self.k_ho            = sym_t("k_ho",            44)
            self.k_wo            = sym_t("k_wo",            48)
            self.k_stride_h      = sym_t("k_stride_h",      52)
            self.k_stride_w      = sym_t("k_stride_w",      56)
            self.k_dilation_h    = sym_t("k_dilation_h",    60)
            self.k_dilation_w    = sym_t("k_dilation_w",    64)
            self.k_pad_h         = sym_t("k_pad_h",         68)
            self.k_pad_w         = sym_t("k_pad_w",         72)
            self.k_y             = sym_t("k_y",             76)
            self.k_x             = sym_t("k_x",             80)
            self.k_dtile_iy      = sym_t("k_dtile_iy",      84)
            self.k_dtile_ix      = sym_t("k_dtile_ix",      88)
            self.k_dtile_dy      = sym_t("k_dtile_dy",      92)
            self.k_dtile_dx      = sym_t("k_dtile_dx",      96)
            self.k_dtile_y       = sym_t("k_dtile_y",       100)
            self.k_dtile_x       = sym_t("k_dtile_x",       104)
            self.k_dtile_h       = sym_t("k_dtile_h",       108)
            self.k_dtile_w       = sym_t("k_dtile_w",       112)
            self.k_dslice_y      = sym_t("k_dslice_y",      116)
            self.k_dslice_x      = sym_t("k_dslice_x",      120)
            self.k_dslice_h      = sym_t("k_dslice_h",      124)
            self.k_dslice_w      = sym_t("k_dslice_w",      128)
            self.k_dslice_h_left = sym_t("k_dslice_h_left", 132)
            self.k_dslice_w_left = sym_t("k_dslice_w_left", 136)
            self.k_pack0         = sym_t("k_pack0",         140)
            self.k_end           = sym_t("k_end",           144)

        def get_count(self):
            return self.k_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('k_'):
                    self._emit(v.declare())

    class kernel_sgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            self.outer = outer

            self.s_ka                      = sym_t("s_ka"                     ,0)
            self.s_bx                      = sym_t("s_bx"                     ,2)
            self.s_p_in                    = sym_t("s_p_in"                   ,4)
            self.s_p_wei                   = sym_t("s_p_wei"                  ,8)
            self.s_p_out                   = sym_t("s_p_out"                  ,12)
            self.s_hi                      = sym_t("s_hi"                     ,16)
            self.s_wi                      = sym_t("s_wi"                     ,17)
            self.s_n                       = sym_t("s_n"                      ,18)
            self.s_k                       = sym_t("s_k"                      ,19)
            self.s_c                       = sym_t("s_c"                      ,20)
            if outer.tunable.nxe != 0:
                self.s_ho                  = sym_t("s_ho"                     ,21)
                self.s_wo                  = sym_t("s_wo"                     ,22)
                self.s_stride_h            = sym_t("s_stride_h"               ,23)
                self.s_stride_w            = sym_t("s_stride_w"               ,24)
                self.s_dilation_h          = sym_t("s_dilation_h"             ,25)
                self.s_dilation_w          = sym_t("s_dilation_w"             ,26)
                self.s_pad_h               = sym_t("s_pad_h"                  ,27)
                self.s_pad_w               = sym_t("s_pad_w"                  ,28)
                self.s_y                   = sym_t("s_y"                      ,29)
                self.s_x                   = sym_t("s_x"                      ,30)
                self.s_dtile_iy            = sym_t("s_dtile_iy"               ,31)
                self.s_dtile_ix            = sym_t("s_dtile_ix"               ,32)
                self.s_dtile_dy            = sym_t("s_dtile_dy"               ,33)
                self.s_dtile_dx            = sym_t("s_dtile_dx"               ,34)
                self.s_dtile_y             = sym_t("s_dtile_y"                ,35)
                self.s_dtile_x             = sym_t("s_dtile_x"                ,36)
                self.s_dtile_h             = sym_t("s_dtile_h"                ,37)
                self.s_dtile_w             = sym_t("s_dtile_w"                ,38)
                self.s_dslice_y            = sym_t("s_dslice_y"               ,39)
                self.s_dslice_x            = sym_t("s_dslice_x"               ,40)
                self.s_dslice_h            = sym_t("s_dslice_h"               ,41)
                self.s_dslice_w            = sym_t("s_dslice_w"               ,42)
                self.s_dslice_h_left       = sym_t("s_dslice_h_left"          ,43)
                self.s_dslice_w_left       = sym_t("s_dslice_w_left"          ,44)
                sseq                       = gpr_sequencer_t(44 + 1)
            else:
                sseq                       = gpr_sequencer_t(20 + 1)

            self.s_out_stride_k            = sym_t("s_out_stride_k"           ,sseq(1))
            if outer.tunable.nxe == 0:
                self.s_stride_hw           = sym_t("s_stride_hw"              ,sseq(1))
            self.s_out_stride_k0           = sym_t("s_out_stride_k0"          ,sseq(1))
            self.s_out_stride_n            = sym_t("s_out_stride_n"           ,sseq(1))
            self.s_out_stride_n0           = sym_t("s_out_stride_n0"          ,sseq(1))

            if outer.tunable.gemm_m_unmerge_cluster == 1:
                self.s_in_stride_c0        = sym_t("s_in_stride_c0"           ,sseq(1))
            self.s_in_stride_c             = sym_t("s_in_stride_c"            ,sseq(1))
            if outer.tunable.gemm_n_unmerge_cluster == 1:
                self.s_in_stride_n0        = sym_t("s_in_stride_n0"           ,sseq(1))
            self.s_in_stride_n             = sym_t("s_in_stride_n"            ,sseq(1))

            if outer.tunable.nxe != 0:
                self.s_wei_stride_c        = sym_t("s_wei_stride_c"           ,sseq(1))
            self.s_wei_stride_c0           = sym_t("s_wei_stride_c0"          ,sseq(1))
            self.s_wei_stride_k            = sym_t("s_wei_stride_k"           ,sseq(1))
            self.s_wei_stride_k0           = sym_t("s_wei_stride_k0"          ,sseq(1))

            if outer.tunable.nxe != 0:
                self.s_stride_dslice_hw    = sym_t("s_stride_dslice_hw"       ,sseq(1))
                self.s_stride_dslice_yx    = sym_t("s_stride_dslice_yx"       ,sseq(1))

            if outer.tunable.nxe != 0:
                self.s_out_stride_k_k1         = sym_t("s_out_stride_k_k1"        ,self.s_stride_h.value)
                self.s_out_stride_k_k0_k1_diff = sym_t("s_out_stride_k_k0_k1_diff",self.s_stride_w.value)
                self.s_wei_stride_k_k1         = sym_t("s_wei_stride_k_k1"        ,self.s_dilation_h.value)
                self.s_wei_stride_k_k0_k1_diff = sym_t("s_wei_stride_k_k0_k1_diff",self.s_dilation_w.value)
            else:
                self.s_out_stride_k_k1         = sym_t("s_out_stride_k_k1"        ,sseq(1))
                self.s_out_stride_k_k0_k1_diff = sym_t("s_out_stride_k_k0_k1_diff",sseq(1))
                self.s_wei_stride_k_k1         = sym_t("s_wei_stride_k_k1"        ,sseq(1))
                self.s_wei_stride_k_k0_k1_diff = sym_t("s_wei_stride_k_k0_k1_diff",sseq(1))

            self.s_move_slice_k_k1         = sym_t("s_move_slice_k_k1"        ,sseq(1))
            if outer.tunable.nxe != 0:
                self.s_move_slice_k_dsy    = sym_t("s_move_slice_k_dsy"       ,self.s_dslice_h_left.value)
                self.s_move_slice_k_dsx    = sym_t("s_move_slice_k_dsx"       ,self.s_dslice_w_left.value)

            self.s_block_gtc_ib            = sym_t("s_block_gtc_ib"           ,sseq(1))
            #if outer.tunable.gemm_m_unmerge_cluster == 0:
            self.s_block_gtc_ic            = sym_t("s_block_gtc_ic"           ,sseq(1))
            #else:
            #    self.s_block_gtc_ic0           = sym_t("s_block_gtc_ic0"           ,sseq(1))
            #    self.s_block_gtc_ic1           = sym_t("s_block_gtc_ic1"           ,sseq(1))
            self.s_block_gtc_in0           = sym_t("s_block_gtc_in0"          ,sseq(1))
            self.s_block_gtc_in1b          = sym_t("s_block_gtc_in1b"         ,sseq(1))

            self.s_knum                    = sym_t("s_knum"                   ,1)
            self.s_gemm_k_num_k1           = sym_t("s_gemm_k_num_k1"          ,2)
            if outer.tunable.nxe != 0:
                self.s_gemm_k_num_dsy      = sym_t("s_gemm_k_num_dsy"         ,self.s_dslice_y.value)
                self.s_gemm_k_num_dsx      = sym_t("s_gemm_k_num_dsx"         ,self.s_dslice_x.value)
                self.s_dtile_dy_neg        = sym_t("s_dtile_dy_neg"           ,self.s_dtile_dy.value)
                self.s_dtile_dx_neg        = sym_t("s_dtile_dx_neg"           ,self.s_dtile_dx.value)

            self.s_kitr                    = sym_t("s_kitr"                   ,3)
            if outer.tunable.precache_soffset:
                m_out_2d_global_load, m_wei_2d_global_load = outer.get_macro_global_load()
                out_npc = m_out_2d_global_load.get_num_precache_soffset()
                wei_npc = m_wei_2d_global_load.get_num_precache_soffset()
                self.s_out_offset          = sym_t("s_out_offset"             ,sseq(out_npc))   # if this number is zero, it is also OK, since we would not use
                self.s_wei_offset          = sym_t("s_wei_offset"             ,sseq(wei_npc))
            self.s_tmp                     = sym_t("s_tmp"                    ,sseq(6, 2))
            self.s_end                     = sym_t("s_end"                    ,sseq())

        def get_count(self):
            return self.s_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('s_'):
                    self._emit(v.declare())

    class kernel_vgpr_t(mc_base_t):
        def __init__(self, mc, outer):
            mc_base_t.__init__(self, mc)
            vseq = gpr_sequencer_t()
            self.outer               = outer
            self.v_c                 = sym_t("v_c"            ,vseq(outer.tunable.num_vgpr_accumulate_c))
            v_c_num                  = vseq()
            self.v_a                 = sym_t("v_a"            ,vseq(outer.tunable.num_vgpr_accumulate_a))
            self.v_b                 = sym_t("v_b"            ,vseq(outer.tunable.num_vgpr_accumulate_b))
            self.v_gld_a             = sym_t("v_gld_a"        ,vseq(outer.tunable.num_vgpr_global_load_a))
            self.v_gld_b             = sym_t("v_gld_b"        ,vseq(outer.tunable.num_vgpr_global_load_b))
            self.v_sst_a_os          = sym_t("v_sst_a_os"     ,vseq(1))
            self.v_sst_b_os          = sym_t("v_sst_b_os"     ,vseq(1))
            self.v_sld_a_os          = sym_t("v_sld_a_os"     ,vseq(1))
            self.v_sld_b_os          = sym_t("v_sld_b_os"     ,vseq(1))
            self.v_out_iho           = sym_t("v_out_iho"      ,vseq(1))
            self.v_out_iwo           = sym_t("v_out_iwo"      ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_out_dslice_ih     = sym_t("v_out_dslice_ih",vseq(1))
                self.v_out_dslice_iw     = sym_t("v_out_dslice_iw",vseq(1))
            self.v_out_os            = sym_t("v_out_os"       ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_out_os_base   = sym_t("v_out_os_base"  ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_wei_iy        = sym_t("v_wei_iy"       ,vseq(1))
                self.v_wei_ix        = sym_t("v_wei_ix"       ,vseq(1))
                self.v_dtile_iy      = sym_t("v_dtile_iy"     ,vseq(1))
                self.v_dtile_ix      = sym_t("v_dtile_ix"     ,vseq(1))
            self.v_wei_os            = sym_t("v_wei_os"       ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_wei_os_base   = sym_t("v_wei_os_base"  ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_out_flag      = sym_t("v_out_flag"     ,vseq(1))
            self.v_co_sst            = sym_t("v_co_sst"       ,vseq(1))
            self.v_co_sld            = sym_t("v_co_sld"       ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_in_flag       = sym_t("v_in_flag"      ,vseq(1))
            self.v_in_os             = sym_t("v_in_os"        ,vseq(1))
            self.v_gtc_ik1           = sym_t("v_gtc_ik1"      ,vseq(1))
            if outer.tunable.nxe != 0:
                self.v_gtc_dslice_iy = sym_t("v_gtc_dslice_iy",vseq(1))
                self.v_gtc_dslice_ix = sym_t("v_gtc_dslice_ix",vseq(1))
            self.v_move_slice_k_ik1  = sym_t("v_move_slice_k_ik1" , self.v_gtc_ik1.value)
            if outer.tunable.nxe != 0:
                self.v_move_slice_k_idsy = sym_t("v_move_slice_k_idsy", self.v_gtc_dslice_iy.value)
                self.v_move_slice_k_idsx = sym_t("v_move_slice_k_idsx", self.v_gtc_dslice_ix.value)

            self.v_gtc_ic0       = sym_t("v_gtc_ic0"      ,v_c_num - 1)
            self.v_gtc_ic1       = sym_t("v_gtc_ic1"      ,v_c_num - 2)
            self.v_gtc_ik0       = sym_t("v_gtc_ik0"      ,v_c_num - 3)
            self.v_gtc_ik1e      = sym_t("v_gtc_ik1e"     ,v_c_num - 4)

            self.v_gtc_in0       = sym_t("v_gtc_in0"      ,v_c_num - 8)
            self.v_gtc_in1b      = sym_t("v_gtc_in1b"     ,v_c_num - 9)
            self.v_gtc_in1       = sym_t("v_gtc_in1"      ,v_c_num - 10)
            self.v_gemm_in       = sym_t("v_gemm_in"      ,v_c_num - 11)
            self.v_gemm_im       = sym_t("v_gemm_im"      ,v_c_num - 12)

            if v_c_num < 16:
                self.v_in_in0        = sym_t("v_in_in0"       ,vseq(1))
                self.v_in_in1b       = sym_t("v_in_in1b"      ,vseq(1))
                self.v_in_in1        = sym_t("v_in_in1"       ,vseq(1))
                self.v_in_ihi        = sym_t("v_in_ihi"       ,vseq(1))
                self.v_in_iwi        = sym_t("v_in_iwi"       ,vseq(1))
                if outer.tunable.nxe != 0:
                    self.v_in_dslice_ih  = sym_t("v_in_dslice_ih" ,vseq(1))
                    self.v_in_dslice_iw  = sym_t("v_in_dslice_iw" ,vseq(1))
                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,vseq(1))
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,vseq(1))
                else:
                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,vseq(1))
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,vseq(1))
            else:
                self.v_in_in0        = sym_t("v_in_in0"       ,v_c_num - 13)
                self.v_in_in1b       = sym_t("v_in_in1b"      ,v_c_num - 14)
                self.v_in_in1        = sym_t("v_in_in1"       ,v_c_num - 15)
                self.v_in_ihi        = sym_t("v_in_ihi"       ,v_c_num - 16)
                self.v_in_iwi        = sym_t("v_in_iwi"       ,v_c_num - 17)
                if outer.tunable.nxe != 0:
                    self.v_in_dslice_ih  = sym_t("v_in_dslice_ih" ,v_c_num - 18)
                    self.v_in_dslice_iw  = sym_t("v_in_dslice_iw" ,v_c_num - 19)
                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,v_c_num - 20)
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,v_c_num - 21)
                else:
                    self.v_co_sub_m_index = sym_t("v_co_sub_m_index" ,v_c_num - 18)
                    self.v_co_sub_n_index = sym_t("v_co_sub_n_index" ,v_c_num - 19)

            self.v_tmp           = sym_t("v_tmp"          ,vseq(6, 2))
            self.v_end           = sym_t("v_end"          ,vseq())

        def get_count(self):
            return self.v_end.value

        def emit(self):
            for k, v in self.__dict__.items():
                if k.startswith('v_'):
                    self._emit(v.declare())


    def get_thread_lengths(self):
        t_ta = self.tunable.tensor_a_thread_lengths
        t_tb = self.tunable.tensor_b_thread_lengths

        assert len(t_ta) == 4 and len(t_tb) == 4

        t_k0, t_k1e, t_c0, t_c1  = t_ta[0], t_ta[1], t_ta[2], t_ta[3]
        _   ,    _,  t_n0, t_n1b = t_tb[0], t_tb[1], t_tb[2], t_tb[3]

        return t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b # M, K, N


    def get_cluster_lengths(self):
        c_ta = self.tunable.tensor_a_cluster_lengths
        c_tb = self.tunable.tensor_b_cluster_lengths

        assert len(c_ta) == 4 and len(c_tb) == 4

        c_k0, c_k1e, c_c0, c_c1  = c_ta[0], c_ta[1], c_ta[2], c_ta[3]
        _   ,     _, c_n0, c_n1b = c_tb[0], c_tb[1], c_tb[2], c_tb[3]

        return c_c0, c_c1, c_k0, c_k1e, c_n0, c_n1b # M, K, N

    def get_dims_lengths(self):
        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        c_c0, c_c1, c_k0, c_k1e, c_n0, c_n1b = self.get_cluster_lengths()

        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = \
                t_c0*c_c0, t_c1*c_c1, t_k0*c_k0, t_k1e*c_k1e, t_n0*c_n0, t_n1b*c_n1b

        return n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b

    def get_thread_copy_dims(self):
        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        out_thread_copy_dims    = [t_k0, t_k1e, t_n0, t_n1b]
        wei_thread_copy_dims    = [t_k0, t_k1e, t_c0, t_c1]
        return out_thread_copy_dims, wei_thread_copy_dims

    def get_thread_copy_index(self):
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index   = _find_non_1_index_in_list(out_thread_copy_dims)
        wei_thread_copy_index   = _find_non_1_index_in_list(wei_thread_copy_dims)
        #assert len(out_thread_copy_index) in (1, 2) and len(wei_thread_copy_index) in (1, 2),\
        #        f'out_thread_copy_dims:{out_thread_copy_dims} wei_thread_copy_dims:{wei_thread_copy_dims}'
        return out_thread_copy_index, wei_thread_copy_index

    def get_macro_global_load(self):
        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()

        ctrl_out_gld = ctrl_2d_global_load_t()
        ctrl_wei_gld = ctrl_2d_global_load_t()

        ctrl_out_gld.vector_d1 = igemm_gcd(t_n1b, 4) if t_n1b != 1 else 1
        ctrl_wei_gld.vector_d1 = igemm_gcd(t_c1, 4) if self.tunable.nxe == 0 else 1

        if self.out_thread_copy_ndim == 2:
            ctrl_out_gld.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            #if t_n0 != 1 and t_n1b == 1:
            #    ctrl_out_gld.src_order = 1              # this reorder seems have little impact...
        elif self.out_thread_copy_ndim == 1:
            ctrl_out_gld.length_d0 = 1
            ctrl_out_gld.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
        else:
            assert False

        if self.wei_thread_copy_ndim == 2:
            ctrl_wei_gld.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
        elif self.wei_thread_copy_ndim == 1:
            ctrl_wei_gld.length_d0 = 1
            ctrl_wei_gld.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
        else:
            assert False

        if self.tunable.precache_soffset:
            return macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_out_gld), \
                    macro_igemm_2d_global_load_precache_soffset_t(self.mc, ctrl_wei_gld)
        else:
            return macro_igemm_2d_global_load_t(self.mc, ctrl_out_gld),  macro_igemm_2d_global_load_t(self.mc, ctrl_wei_gld)

    def get_macro_global_store(self):
        return macro_igemm_write_4d_strided_t(self.mc)

    def get_macro_shared_store(self):
        out_thread_copy_dims, wei_thread_copy_dims = self.get_thread_copy_dims()
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()
        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()

        if gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            # t_k0, t_k1e, t_n0, t_n1b
            out_stride_list = [n_k1e*n_n0*n_n1b, n_n0*n_n1b, n_n1b, 1]
        else:
            # t_k0, t_k1e, t_n0, t_n1b
            out_stride_list = [n_k1e*n_n0*n_n1b, n_n0*n_n1b, 1, n_n0]


        if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1:
            wei_stride_list = [n_k1e*n_c0*n_c1, n_c0*n_c1, n_c1, 1]
        else:
            wei_stride_list = [n_k1e*n_c0*n_c1, n_c0*n_c1, 1, n_c0]

        out_sst_ctrl = ctrl_2d_shared_store_t()
        wei_sst_ctrl = ctrl_2d_shared_store_t()

        if self.out_thread_copy_ndim == 2:
            out_sst_ctrl.length_d0 = out_thread_copy_dims[out_thread_copy_index[0]]
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            if gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
                out_sst_ctrl.vector_d1 = t_n1b
            else:
                out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[1]]
            #out_sst_ctrl.vector_d1 = t_n1b
            out_sst_ctrl.stride_d0 = out_stride_list[out_thread_copy_index[0]] * data_byte
            out_sst_ctrl.stride_d1 = out_stride_list[out_thread_copy_index[1]] * data_byte
            #out_sst_ctrl.stride_d1 = 1
        elif self.out_thread_copy_ndim == 1:
            out_sst_ctrl.length_d0 = 1
            out_sst_ctrl.length_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            if (gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B and t_n1b != 1) or \
                (gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N1B_N0 and t_n0 != 1):
                out_sst_ctrl.vector_d1 = out_thread_copy_dims[out_thread_copy_index[0]]
            else:
                out_sst_ctrl.vector_d1 = 1
            out_sst_ctrl.stride_d0 = 1
            out_sst_ctrl.stride_d1 = out_stride_list[out_thread_copy_index[0]] * data_byte
            if out_sst_ctrl.length_d1 == 8 and out_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                out_sst_ctrl.length_d0 = 2
                out_sst_ctrl.length_d1 = 4
                out_sst_ctrl.vector_d1 = 4
                out_sst_ctrl.stride_d0 = 4 * data_byte
        else:
            assert False

        if self.wei_thread_copy_ndim == 2:
            wei_sst_ctrl.length_d0 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1:
                wei_sst_ctrl.vector_d1 = t_c1
            else:
                wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[1]]
            #wei_sst_ctrl.vector_d1 = t_c1
            wei_sst_ctrl.stride_d0 = wei_stride_list[wei_thread_copy_index[0]] * data_byte
            wei_sst_ctrl.stride_d1 = wei_stride_list[wei_thread_copy_index[1]] * data_byte
        elif self.wei_thread_copy_ndim == 1:
            wei_sst_ctrl.length_d0 = 1
            wei_sst_ctrl.length_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]

            if (gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1 and t_c1 != 1) or \
                (gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C1_C0 and t_c0 != 1):
                wei_sst_ctrl.vector_d1 = wei_thread_copy_dims[wei_thread_copy_index[0]]
            else:
                wei_sst_ctrl.vector_d1 = 1

            wei_sst_ctrl.stride_d0 = 1
            wei_sst_ctrl.stride_d1 = wei_stride_list[wei_thread_copy_index[0]] * data_byte
            if wei_sst_ctrl.length_d1 == 8 and wei_sst_ctrl.vector_d1 != 1:
                # assert False
                # TODO: this is indeed not optimal. may consider shuffle in the future.
                wei_sst_ctrl.length_d0 = 2
                wei_sst_ctrl.length_d1 = 4
                wei_sst_ctrl.vector_d1 = 4
                wei_sst_ctrl.stride_d0 = 4 * data_byte
        else:
            assert False

        # print(f"out_sst_ctrl.vector_d1:{out_sst_ctrl.vector_d1}, wei_sst_ctrl.vector_d1:{wei_sst_ctrl.vector_d1}")

        return macro_igemm_2d_shared_store_t(self.mc, out_sst_ctrl), macro_igemm_2d_shared_store_t(self.mc, wei_sst_ctrl)

    def get_macro_shared_load(self):
        return None

    def get_macro_out_update_os(self):
        return macro_igemm_bwd_gtc_out_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision))
    def get_macro_out_update_hw(self):
        if self.tunable.nxe != 0:
            return macro_igemm_bwd_gtc_out_update_hw_t(self.mc)
        return None
    def get_macro_wei_update_os(self):
        return macro_igemm_bwd_gtc_wei_update_os_t(self.mc, amdgpu_precision_data_byte(self.tunable.precision))
    def get_macro_wei_update_yx(self):
        if self.tunable.nxe != 0:
            return macro_igemm_bwd_gtc_wei_update_yx_t(self.mc)
        return None
    def get_macro_set_flag_hw(self):
        return macro_igemm_bwd_gtc_set_flag_hw(self.mc)

    def get_macro_move_slice_window(self):
        if self.tunable.nxe != 0:
            return macro_igemm_bwd_gtc_move_slice_window_k_dsy_dsx(self.mc, self.tunable)
        else:
            if self.is_1d_move_slice_k():
                return macro_igemm_bwd_gtc_move_slice_window_k_1d(self.mc, self.tunable)
            else:
                return macro_igemm_bwd_gtc_move_slice_window_k(self.mc, self.tunable)

    def get_symbol_global_load_s_stride_d0_d1(self):
        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        # get the symbol object that load 2d may use
        s = self.sgpr
        s_dummy = sym_t("s_dummy")
        out_thread_copy_index, wei_thread_copy_index = self.get_thread_copy_index()
        out_stride_gprs = [s.s_out_stride_k0 if t_k0 != 1 else s_dummy,
                    s_dummy if self.tunable.nxe != 0 else s.s_out_stride_k,
                    s.s_out_stride_n0 if t_n0 != 1 else s_dummy,
                    s_dummy]
        wei_stride_gprs = [s.s_wei_stride_k0 if t_k0 != 1 else s_dummy,
                    s_dummy if self.tunable.nxe != 0 else s.s_wei_stride_k,
                    s.s_wei_stride_c0 if t_c0 != 1 else s_dummy,
                    s.s_wei_stride_c if self.tunable.nxe != 0 else s_dummy]

        if self.out_thread_copy_ndim == 2:
            s_out_stride_d0 = out_stride_gprs[out_thread_copy_index[0]]
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[1]]
        elif self.out_thread_copy_ndim == 1:
            s_out_stride_d0 = s_dummy
            s_out_stride_d1 = out_stride_gprs[out_thread_copy_index[0]]
        else:
            assert False

        if self.wei_thread_copy_ndim == 2:
            s_wei_stride_d0 = wei_stride_gprs[wei_thread_copy_index[0]]
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[1]]
        elif self.wei_thread_copy_ndim == 1:
            s_wei_stride_d0 = s_dummy
            s_wei_stride_d1 = wei_stride_gprs[wei_thread_copy_index[0]]
        else:
            assert False

        return s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1

    def get_kernel_code(self):
        kernel_code = amdgpu_kernel_code_t({
                'enable_sgpr_kernarg_segment_ptr'   :   1,
                'enable_sgpr_workgroup_id_x'        :   1,
                'enable_vgpr_workitem_id'           :   0,
                'workgroup_group_segment_byte_size' :   self.tunable.lds_total,
                'kernarg_segment_byte_size'         :   self.karg.get_count(),
                'wavefront_sgpr_count'              :   self.sgpr.get_count() + 2*3,
                'workitem_vgpr_count'               :   self.vgpr.get_count()
                })
        return kernel_code

    def get_kernel_args(self):
        '''
            float *p_in;
            float *p_wei;
            float *p_out;
            int hi;
            int wi;
            int n;
            int k;
            int c;
            int ho;
            int wo;
            int stride_h;
            int stride_w;
            int dilation_h;
            int dilation_w;
            int pad_h;
            int pad_w;
            int y;
            int x;
            int dtile_iy;
            int dtile_ix;
            int dtile_dy;
            int dtile_dx;
            int dtile_y;
            int dtile_x;
            int dtile_h;
            int dtile_w;
            int dslice_y;
            int dslice_x;
            int dslice_h;
            int dslice_w;
            int dslice_h_left;
            int dslice_w_left;
            int __pack0;
        '''
        kas = []
        # name: {}, .size: {}, .offset: {}, .value_kind: {}, .value_type
        kas.append(amdgpu_kernel_arg_t('p_in'          , 8,   0, 'global_buffer','f32',address_space='global',is_const='false'))
        kas.append(amdgpu_kernel_arg_t('p_wei'         , 8,   8, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('p_out'         , 8,  16, 'global_buffer','f32',address_space='global',is_const='true'))
        kas.append(amdgpu_kernel_arg_t('hi'            , 4,  24, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wi'            , 4,  28, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('n'             , 4,  32, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('k'             , 4,  36, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('c'             , 4,  40, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('ho'            , 4,  44, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('wo'            , 4,  48, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_h'      , 4,  52, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('stride_w'      , 4,  56, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_h'    , 4,  60, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dilation_w'    , 4,  64, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_h'         , 4,  68, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('pad_w'         , 4,  72, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('y'             , 4,  76, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('x'             , 4,  80, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_iy'      , 4,  84, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_ix'      , 4,  88, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_dy'      , 4,  92, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_dx'      , 4,  96, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_y'       , 4, 100, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_x'       , 4, 104, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_h'       , 4, 108, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dtile_w'       , 4, 112, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_y'      , 4, 116, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_x'      , 4, 120, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_h'      , 4, 124, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_w'      , 4, 128, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_h_left' , 4, 132, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('dslice_w_left' , 4, 136, 'by_value', 'i32'))
        kas.append(amdgpu_kernel_arg_t('__pack0'       , 4, 140, 'by_value', 'i32'))
        return kas


    def get_kernel_info(self):
        kernel_code = self.get_kernel_code()
        kernel_args = self.get_kernel_args()
        kernel_info = amdgpu_kernel_info_t(kernel_code, self.name(), self.tunable.block_size, kernel_args)
        return kernel_info

    def get_kernel_macros(self):
        kernel_macros = []
        for attrs in dir(self):
            if attrs.startswith('get_macro_'):
                functor = getattr(self, attrs)
                rtn = functor()
                if rtn is None:
                    continue
                if type(rtn) is tuple:
                    kernel_macros.extend([m for m in rtn])
                else:
                    kernel_macros.append(rtn)
        return kernel_macros


    def emit_kernel_prologue(self):
        s = self.sgpr
        v = self.vgpr
        k = self.karg
        gemm_m_unmerge_cluster = self.tunable.gemm_m_unmerge_cluster
        gemm_n_unmerge_cluster = self.tunable.gemm_n_unmerge_cluster
        gemm_k_unmerge_cluster = self.tunable.gemm_k_unmerge_cluster

        t_c0, t_c1, t_k0, t_k1e, t_n0, t_n1b = self.get_thread_lengths()
        c_c0, c_c1, c_k0, c_k1e, c_n0, c_n1b = self.get_cluster_lengths()
        n_c0, n_c1, n_k0, n_k1e, n_n0, n_n1b = self.get_dims_lengths()

        unmerge_sub_n = self.tunable.unmerge_sub_n
        if gemm_n_unmerge_cluster == 0:
            assert unmerge_sub_n % n_n0 == 0, f"unmerge_sub_n:{unmerge_sub_n}, n_n0:{n_n0}"
            unmerge_sub_n1 = unmerge_sub_n // n_n0
            assert n_n1b % unmerge_sub_n1 == 0, f"n_n1b:{n_n1b}, unmerge_sub_n1:{unmerge_sub_n1}"
        elif gemm_n_unmerge_cluster == 1:
            assert c_n0 == 1 and c_n1b != 1 and t_n0 != 1 and t_n1b == 1, "current implementation only support this stratagy"
            unmerge_sub_n1 = unmerge_sub_n
        else:
            assert False, f"unsupported gemm_n_unmerge_cluster:{self.tunable.gemm_n_unmerge_cluster}"

        unmerge_sub_k = self.tunable.unmerge_sub_k
        if gemm_k_unmerge_cluster == 0:
            assert unmerge_sub_k % n_k0 == 0, f"unmerge_sub_k:{unmerge_sub_k}, n_k0:{n_k0}"
            unmerge_sub_k1 = unmerge_sub_k // n_k0
            assert n_k1e % unmerge_sub_k1 == 0, f"n_k1e:{n_k1e}, unmerge_sub_k1:{unmerge_sub_k1}"
        elif gemm_k_unmerge_cluster == 1:
            assert c_k0 == 1 and c_k1e != 1 and t_k0 != 1 and t_k1e == 1, "current implementation only support this stratagy"
            unmerge_sub_k1 = unmerge_sub_k
        else:
            assert False, f"unsupported gemm_k_unmerge_cluster:{self.tunable.gemm_k_unmerge_cluster}"

        if gemm_m_unmerge_cluster == 1:
            assert c_c0 == 1 and c_c1 != 1 and t_c0 != 1 and t_c1 == 1, "current implementation only support this stratagy"

        #assert c_n0 == 1 and c_k0 == 1 and c_c0 == 1, "cluster lengths has no meaning to deal with x0"

        data_byte = amdgpu_precision_data_byte(self.tunable.precision)

        m_out_update_os   = self.get_macro_out_update_os()
        m_out_update_hw   = self.get_macro_out_update_hw()
        m_wei_update_os   = self.get_macro_wei_update_os()
        m_wei_update_yx   = self.get_macro_wei_update_yx()
        m_set_flag_hw     = self.get_macro_set_flag_hw()

        m_out_2d_global_load, m_wei_2d_global_load = self.get_macro_global_load()
        s_out_stride_d0, s_out_stride_d1, s_wei_stride_d0, s_wei_stride_d1 = self.get_symbol_global_load_s_stride_d0_d1()

        tc_index_dispatcher = igemm_thread_cluster_index_dispatcher_t(self.mc)
        tc_index_accumulator = igemm_thread_cluster_index_accumulator_t(self.mc)

        m_int_div_rem_vv = macro_int_div_rem_vv_t(self.mc)
        m_int_div_rem_vs = macro_int_div_rem_vs_t(self.mc)
        m_int_div_rem_ss = macro_int_div_rem_ss_t(self.mc)
        gemm_m_order, gemm_n_order = self.get_lds_gemm_m_gemm_n_order()
        s_dummy = sym_t("s_dummy")

        self._emit(f"; unmerge_sub_k:{unmerge_sub_k}, unmerge_sub_k1:{unmerge_sub_k1}, unmerge_sub_n:{unmerge_sub_n}, unmerge_sub_n1:{unmerge_sub_n1}")
        self._emit(f"; gemm_m_unmerge_cluster:{gemm_m_unmerge_cluster}, gemm_n_unmerge_cluster:{gemm_n_unmerge_cluster}, gemm_k_unmerge_cluster:{gemm_k_unmerge_cluster}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_in((0,1))}],       s[{s.s_ka((0, 1))}],    0+{k.k_p_in()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_wei((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_wei()}")
        self._emit(f"s_load_dwordx2  s[{s.s_p_out((0,1))}],      s[{s.s_ka((0, 1))}],    0+{k.k_p_out()}")
        if self.tunable.nxe != 0:
            self._emit(f"s_load_dwordx16 s[{s.s_hi((0,15))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dwordx8  s[{s.s_dtile_ix((0,7))}],   s[{s.s_ka((0, 1))}],    0+{k.k_dtile_ix()}")
            self._emit(f"s_load_dwordx4  s[{s.s_dslice_x((0,3))}],   s[{s.s_ka((0, 1))}],    0+{k.k_dslice_x()}")
            self._emit(f"s_load_dword    s[{s.s_dslice_w_left()}],   s[{s.s_ka((0, 1))}],    0+{k.k_dslice_w_left()}")
        else:
            self._emit(f"s_load_dwordx4 s[{s.s_hi((0,3))}],        s[{s.s_ka((0, 1))}],    0+{k.k_hi()}")
            self._emit(f"s_load_dword s[{s.s_c()}], s[{s.s_ka((0, 1))}],    0+{k.k_c()}")

        self._emit(f"; output, thread(k0,k1e,n0,n1b): {t_k0}x{t_k1e}x{t_n0}x{t_n1b}, cluster(k0,k1e,n0,n1b): {c_k0}x{c_k1e}x{c_n0}x{c_n1b}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_in1b(),  v.v_tmp(), c_n1b, t_n1b))      # merged dimension no need to do shift per thread here, do shift later
        self._emit(tc_index_dispatcher(v.v_gtc_in0(),   v.v_tmp(), c_n0,  t_n0))
        self._emit(tc_index_dispatcher(v.v_gtc_ik1e(),  v.v_tmp(), c_k1e, t_k1e))      # merged dimension no need to do shift per thread here, do shift later
        self._emit(tc_index_dispatcher(v.v_gtc_ik0(),   v.v_tmp(), c_k0,  t_k0, True))
        self._emit_empty_line()
        self._emit(f"; wei, thread(k0,k1e,c0,c1): {t_k0}x{t_k1e}x{t_c0}x{t_c1}, cluster(k0,k1e,c0,c1) {c_k0}x{c_k1e}x{c_c0}x{c_c1}")
        self._emit(f"v_mov_b32 v[{v.v_tmp()}], v0")
        self._emit(tc_index_dispatcher(v.v_gtc_ic1(), v.v_tmp(), c_c1, t_c1))
        self._emit(tc_index_dispatcher(v.v_gtc_ic0(), v.v_tmp(), c_c0, t_c0, True))
        self._emit_empty_line()
        self._emit(f"s_mov_b32 s[{s.s_p_in(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_in(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_wei(3)}], 0x27000")
        self._emit(f"s_mov_b32 s[{s.s_p_out(2)}], 0xffffffff")
        self._emit(f"s_mov_b32 s[{s.s_p_out(3)}], 0x27000")
        if self.tunable.multihead and self.tunable.nxe != 0:
            self._emit(f"s_mov_b32 s[{s.s_tmp()}], 0xffff")
        self._emit(f"s_waitcnt lgkmcnt(0)")
        self._emit_empty_line()

        if self.tunable.multihead and self.tunable.nxe != 0:
            #label_mh_dispatch_end = f"L_{self.name()}_mh_dispatch_end"
            self._emit(f"; multihead dispatch code start")
            self._emit(f"s_mov_b32 s[0], s[{s.s_dtile_iy()}]        ; normal gridsize. gridsize / num_gemms")
            self._emit(f"s_lshr_b32 s[{s.s_dtile_iy()}], s[{s.s_dtile_ix()}], 16")
            self._emit(f"s_and_b32 s[{s.s_dtile_ix()}], s[{s.s_tmp()}], s[{s.s_dtile_ix()}]")
            self._emit(m_int_div_rem_ss(s.s_tmp(5), '1', s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            #self._emit(m_int_div_rem_ss('1', s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            self._emit(f"; s1:i_y_tilda*i_x_tilda  s_tmp+5: normal s_bx")
            self._emit(f"s_mov_b32 s[{s.s_bx()}], s[{s.s_tmp(5)}]")
            self._emit(m_int_div_rem_ss(s.s_tmp(5), s.s_tmp(4), '1', s.s_dtile_x(), v.v_tmp(5), v.v_tmp(), s.s_tmp()))
            self._emit(f"; s_tmp+4:dtile_iy, s_tmp+5:dtile_ix")
            self._emit(f"s_add_u32 s[{s.s_tmp()}], 1, s[{s.s_tmp(4)}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dtile_iy()}], s[{s.s_tmp()}]  ; (i_y_tilda + 1) * y_dot ")
            self._emit(f"s_add_u32 s[{s.s_tmp(1)}], 1, s[{s.s_tmp(5)}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_dtile_ix()}], s[{s.s_tmp(1)}] ; (i_x_tilda + 1) * x_dot")
            self._emit(f"s_cmp_le_u32 s[{s.s_tmp()}], s[{s.s_y()}]")
            self._emit(f"s_cmov_b32 s[{s.s_dslice_y()}], s[{s.s_dtile_iy()}]")
            self._emit(f"s_cmp_le_u32 s[{s.s_tmp(1)}], s[{s.s_x()}]")
            self._emit(f"s_cmov_b32 s[{s.s_dslice_x()}], s[{s.s_dtile_ix()}]")
            self._emit(f"s_mov_b32 s[{s.s_dtile_iy()}], s[{s.s_tmp(4)}]")
            self._emit(f"s_mov_b32 s[{s.s_dtile_ix()}], s[{s.s_tmp(5)}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_dslice_y()}], s[{s.s_dslice_x()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_k()}], s[{s.s_tmp(1)}]        ; gemm_k, check empty")
            self._emit(f"s_cmp_gt_u32 s[{s.s_tmp()}], 0")
            self._emit(f"s_cbranch_scc0 {self.label_out}        ; early exit if current gemm_k is zero, only happen when filter is 1x1 and have stride or dilation. better not jump")
            self._emit(f"; multihead dispatch code end")
            self._emit_empty_line()

        self._emit(f"; calculate index")

        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_out_stride_k()}],      s[{s.s_ho()}],       s[{s.s_wo()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}],      s[{s.s_k()}],        s[{s.s_out_stride_k()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_c()}],       s[{s.s_hi()}],       s[{s.s_wi()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}],       s[{s.s_c()}],        s[{s.s_in_stride_c()}]")
            if gemm_m_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
                self._emit(f"s_mul_i32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], s[{s.s_tmp()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_c()}],      s[{s.s_y()}],        s[{s.s_x()}]")
            self._emit(f"s_mul_i32 s[{s.s_wei_stride_k()}],      s[{s.s_c()}],        s[{s.s_wei_stride_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_stride_dslice_hw()}],  s[{s.s_dslice_h()}], s[{s.s_dslice_w()}]")
            self._emit(f"s_mul_i32 s[{s.s_stride_dslice_yx()}],  s[{s.s_dslice_y()}], s[{s.s_dslice_x()}]")
            if t_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_out_stride_k0()}], s[{s.s_out_stride_k()}], {igemm_log2(unmerge_sub_k1)}")
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_wei_stride_k()}], {igemm_log2(unmerge_sub_k1)}")
            if t_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], {igemm_log2(unmerge_sub_n1)}")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")
            if t_c0 != 1:
                if gemm_m_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_wei_stride_c0()}], s[{s.s_wei_stride_c()}], {igemm_log2(n_c1)}")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
                    self._emit(f"s_mul_i32 s[{s.s_wei_stride_c0()}], s[{s.s_wei_stride_c()}], s[{s.s_tmp()}]")

            self._emit(f"s_mul_i32 s[{s.s_dtile_dy_neg()}], -1, s[{s.s_dtile_dy()}]")
            self._emit(f"s_mul_i32 s[{s.s_dtile_dx_neg()}], -1, s[{s.s_dtile_dx()}]")
        else:
            self._emit(f"s_mul_i32 s[{s.s_stride_hw()}],         s[{s.s_hi()}],       s[{s.s_wi()}]")
            self._emit(f"s_mov_b32 s[{s.s_out_stride_k()}],       s[{s.s_stride_hw()}]")
            self._emit(f"s_mov_b32 s[{s.s_in_stride_c()}],       s[{s.s_stride_hw()}]")
            if gemm_m_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_c()}], {igemm_log2(n_c0)}")
                self._emit(f"s_mul_i32 s[{s.s_in_stride_c0()}], s[{s.s_in_stride_c()}], s[{s.s_tmp()}]")
            if gemm_n_unmerge_cluster == 1:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_in_stride_n0()}], s[{s.s_in_stride_n()}], s[{s.s_tmp()}]")
            self._emit(f"s_mov_b32 s[{s.s_wei_stride_k()}],      s[{s.s_c()}]")
            self._emit(f"s_mul_i32 s[{s.s_out_stride_n()}],      s[{s.s_k()}],        s[{s.s_stride_hw()}]")
            self._emit(f"s_mul_i32 s[{s.s_in_stride_n()}],       s[{s.s_c()}],        s[{s.s_stride_hw()}]")
            if t_k0 != 1:
                self._emit(f"s_lshl_b32 s[{s.s_out_stride_k0()}], s[{s.s_stride_hw()}], {igemm_log2(unmerge_sub_k1)}")
                self._emit(f"s_lshl_b32 s[{s.s_wei_stride_k0()}], s[{s.s_c()}], {igemm_log2(unmerge_sub_k1)}")
            if t_n0 != 1:
                if gemm_n_unmerge_cluster == 0:
                    self._emit(f"s_lshl_b32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], {igemm_log2(unmerge_sub_n1)}")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                    self._emit(f"s_mul_i32 s[{s.s_out_stride_n0()}], s[{s.s_out_stride_n()}], s[{s.s_tmp()}]")
            if t_c0 != 1:
                if gemm_m_unmerge_cluster == 0:
                    self._emit(f"s_mov_b32 s[{s.s_wei_stride_c0()}], {n_c1}")
                else:
                    self._emit(f"s_lshr_b32 s[{s.s_wei_stride_c0()}], s[{s.s_c()}], {igemm_log2(n_c0)}")

        self._emit(f"; k1e transform")
        if self.tunable.nxe != 0:
            if c_k1e == 1:
                # TODO: this is indeed not wished
                self._emit(f"v_mov_b32 v[{v.v_gtc_ik1()}], 0")
                self._emit(f"v_mov_b32 v[{v.v_gtc_dslice_iy()}], 0")
                self._emit(f"v_mov_b32 v[{v.v_gtc_dslice_ix()}], 0")
            else:
                self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_ik1(), v.v_gtc_ik1e(), s.s_stride_dslice_yx(), v.v_tmp(), s.s_tmp()))
                self._emit(m_int_div_rem_vs(v.v_gtc_dslice_ix(), v.v_gtc_dslice_iy(), v.v_tmp(4), s.s_dslice_x(), v.v_tmp(), s.s_tmp()))
        else:
            self._emit(f"v_mov_b32 v[{v.v_gtc_ik1()}], v[{v.v_gtc_ik1e()}]")

        self._emit_empty_line()
        self._emit(f"; gemm_m_per_block:{self.tunable.gemm_m_per_block}, gemm_n_per_block:{self.tunable.gemm_n_per_block}")
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_stride_dslice_hw()}], s[{s.s_n()}]")
        else:
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_stride_hw()}], s[{s.s_n()}]")
        self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp()}], {igemm_log2(self.tunable.gemm_n_per_block)}")
        self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_tmp(5), s.s_bx(), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        self._emit(f"; s_tmp+4:block_gtc_in, s_tmp+5:block_gtc_im")
        if gemm_m_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block)}")
        else:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_ic()}], s[{s.s_tmp(5)}], {igemm_log2(self.tunable.gemm_m_per_block // n_c0)}")
            #self._emit(f"s_mov_b32 s[{s.s_block_gtc_ic0()}], 0")

        if gemm_n_unmerge_cluster == 0:
            if self.tunable.nxe != 0:
                if unmerge_sub_n1 == 1:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_dslice_hw()}], {igemm_log2(n_n1b)} ; total number of n1b")
                else:
                    if unmerge_sub_n1 == n_n1b:
                        self._emit(f"s_mov_b32 s[0], s[{s.s_stride_dslice_hw()}] ; total number of n1b")
                    else:
                        # self._emit(f"s_lshl_b32 s[{s.s_tmp()}], s[{s.s_stride_dslice_hw()}], {igemm_log2(unmerge_sub_n1)}     ; total number of n1b")
                        # self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp()}], {igemm_log2(n_n1b)}")
                        self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_dslice_hw()}], {igemm_log2(n_n1b // unmerge_sub_n1)}  ; total number of n1b")
            else:
                if unmerge_sub_n1 == 1:
                    self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_hw()}], {igemm_log2(n_n1b)} ; total number of n1b")
                else:
                    if unmerge_sub_n1 == n_n1b:
                        self._emit(f"s_mov_b32 s[0], s[{s.s_stride_hw()}] ; total number of n1b")
                    else:
                        self._emit(f"s_lshr_b32 s[0], s[{s.s_stride_hw()}], {igemm_log2(n_n1b // unmerge_sub_n1)}  ; total number of n1b")

            
        else:
            if self.tunable.nxe != 0:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_stride_dslice_hw()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(n_n1b)}")
            else:
                self._emit(f"s_lshr_b32 s[{s.s_tmp()}], s[{s.s_n()}], {igemm_log2(n_n0)}")
                self._emit(f"s_mul_i32 s[{s.s_tmp(1)}], s[{s.s_stride_hw()}], s[{s.s_tmp()}]")
                self._emit(f"s_lshr_b32 s[0], s[{s.s_tmp(1)}], {igemm_log2(n_n1b)}")

        self._emit(m_int_div_rem_ss(s.s_block_gtc_in1b(), s.s_block_gtc_in0(), s.s_tmp(4), '0', v.v_tmp(5), v.v_tmp(), s.s_tmp()))
        if n_n1b != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_in1b()}], s[{s.s_block_gtc_in1b()}], {igemm_log2(n_n1b)}")
        if n_n0 != 1:
            self._emit(f"s_lshl_b32 s[{s.s_block_gtc_in0()}], s[{s.s_block_gtc_in0()}], {igemm_log2(n_n0)}")

        self._emit_empty_line()

        self._emit(f"; n1b transform")
        if c_n1b == 1:
            self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}]")
        else:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_gtc_in1b()}]")
        if self.tunable.nxe != 0:
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_in1(), v.v_tmp(5), s.s_stride_dslice_hw(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_out_dslice_iw(), v.v_out_dslice_ih(), v.v_tmp(4), s.s_dslice_w(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()
            self._emit(f"; iHTildaLeft, iWTildaLeft")
            self._emit(f"v_add_u32 v[{v.v_out_dslice_ih()}], s[{s.s_dslice_h_left()}], v[{v.v_out_dslice_ih()}]")
            self._emit(f"v_add_u32 v[{v.v_out_dslice_iw()}], s[{s.s_dslice_w_left()}], v[{v.v_out_dslice_iw()}]")
            self._emit(m_out_update_hw(v.v_out_iho(), v.v_out_iwo(), v.v_out_dslice_ih(), v.v_out_dslice_iw(), v.v_gtc_dslice_iy(), v.v_gtc_dslice_ix(), s.s_dtile_dy_neg(), s.s_dtile_dx_neg()))
            self._emit_empty_line()
        else:
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_gtc_in1(), v.v_tmp(5), s.s_stride_hw(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_out_iwo(), v.v_out_iho(),  v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))

        self._emit(f"; calculate output offset")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_in0()}], {igemm_log2(unmerge_sub_n1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_out_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_out_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_out()}], s[{s.s_p_out()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_out(1)}], s[{s.s_p_out(1)}], s[{s.s_tmp(1)}]")
        else:
            pass # no in0
        self._emit_empty_line()

        self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1e, 0, unmerge_sub_k1))
        if self.tunable.nxe != 0:
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_out_stride_k()}], v[{v.v_tmp()}]")
        else:
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_hw()}], v[{v.v_tmp()}]")
        if gemm_n_unmerge_cluster == 0:
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_in0(), v.v_gtc_in1(),c_n0, c_n1b, 0, unmerge_sub_n1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_out_stride_n()}], v[{v.v_tmp(1)}]")
        else:
            # no in0
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_out_stride_n()}], v[{v.v_gtc_in1()}]")

        if self.tunable.nxe != 0:
            self._emit(f"v_add_lshl_u32 v[{v.v_out_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
            self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
        else:
            self._emit(f"v_add_lshl_u32 v[{v.v_tmp(4)}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_out_update_os(v.v_out_os(), v.v_tmp(4), v.v_out_iho(), v.v_out_iwo(), s.s_wi(), v.v_tmp()))
        self._emit_empty_line()

        if self.out_thread_copy_ndim != 1:
            if s_out_stride_d0 != s_dummy:
                #self._emit(f"s_lshl_b32 s[{s_out_stride_d0()}], s[{s_out_stride_d0()}], {igemm_log2(data_byte)}")
                self._emit(self.try_shift_stride(s_out_stride_d0, igemm_log2(data_byte)))
        if s_out_stride_d1 != s_dummy:
            #self._emit(f"s_lshl_b32 s[{s_out_stride_d1()}], s[{s_out_stride_d1()}], {igemm_log2(data_byte)}")
            self._emit(self.try_shift_stride(s_out_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            #assert type(m_out_2d_global_load) is macro_igemm_2d_global_load_precache_soffset_t
            #init_precache_soffset(s_stride_d0, s_stride_d1, s_offset, s_tmp):
            self._emit(m_out_2d_global_load.init_precache_soffset(s_out_stride_d0(), s_out_stride_d1(), s.s_out_offset(), s.s_tmp()))

        # load out
        self._emit(self.global_load_out())
        self._emit_empty_line()

        self._emit(f"; calculate wei offset")
        if self.tunable.nxe != 0:
            self._emit(f"v_mov_b32 v[{v.v_dtile_iy()}], s[{s.s_dtile_iy()}]")
            self._emit(f"v_mov_b32 v[{v.v_dtile_ix()}], s[{s.s_dtile_ix()}]")
            self._emit(m_wei_update_yx(v.v_wei_iy(), v.v_wei_ix(), v.v_gtc_dslice_iy(), v.v_gtc_dslice_ix(), s.s_dtile_y(), s.s_dtile_x(), v.v_dtile_iy(), v.v_dtile_ix()))
            self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ic0(), v.v_gtc_ic1(), c_c0, c_c1, n_c0, n_c1))
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_ic()}], v[{v.v_tmp()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_wei_stride_c()}], v[{v.v_tmp(5)}]")
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1e, 0, unmerge_sub_k1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wei_stride_k()}], v[{v.v_tmp(1)}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os_base()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
            self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
        else:
            self._emit(tc_index_accumulator(v.v_tmp(), v.v_gtc_ic0(), v.v_gtc_ic1(), c_c0, c_c1, n_c0, n_c1))
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_block_gtc_ic()}], v[{v.v_tmp()}] ; c index")
            self._emit(tc_index_accumulator(v.v_tmp(1), v.v_gtc_ik0(), v.v_gtc_ik1(), c_k0, c_k1e, 0, unmerge_sub_k1))
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_c()}], v[{v.v_tmp(1)}]")
            self._emit(f"v_add_lshl_u32 v[{v.v_wei_os()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}], {igemm_log2(data_byte)}")
        self._emit_empty_line()

        if self.wei_thread_copy_ndim != 1:
            if s_wei_stride_d0 != s_dummy:
                #self._emit(f"s_lshl_b32 s[{s_wei_stride_d0()}], s[{s_wei_stride_d0()}], {igemm_log2(data_byte)}")
                self._emit(self.try_shift_stride(s_wei_stride_d0, igemm_log2(data_byte)))
        if s_wei_stride_d1 != s_dummy:
            #self._emit(f"s_lshl_b32 s[{s_wei_stride_d1()}], s[{s_wei_stride_d1()}], {igemm_log2(data_byte)}")
            self._emit(self.try_shift_stride(s_wei_stride_d1, igemm_log2(data_byte)))
        self._emit_empty_line()

        if self.tunable.precache_soffset:
            self._emit(m_wei_2d_global_load.init_precache_soffset(s_wei_stride_d0(), s_wei_stride_d1(), s.s_wei_offset(), s.s_tmp()))

        self._emit(self.global_load_wei())
        self._emit_empty_line()

        self._emit(f"v_mov_b32 v[{v.v_tmp(5)}], v0")
        self._emit(self.thread_mapping(v.v_gemm_in(), v.v_gemm_im(), v.v_tmp(5), v.v_tmp()))

        self._emit(f"; LDS store, out: k0,k1e,n0,n1b: {t_k0}x{t_k1e}x{t_n0}x{t_n1b}, {c_k0}x{c_k1e}x{c_n0}x{c_n1b}, order:{gemm_n_order}")
        if gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            if c_n1b == 1:
                # TODO: remove this path, not possible go here
                assert c_n0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_n1b)},  v[{v.v_gtc_in0()}]")
            else:
                if c_n0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_in1b()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in0()}], {igemm_log2(n_n1b)}, v[{v.v_gtc_in1b()}]")
        else:
            assert t_n0 != 1
            if c_n1b == 1:
                # this is not prefered
                assert c_n0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_in0()}]")
            else:
                if c_n0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_n0)}, v[{v.v_gtc_in1b()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_in1b()}], {igemm_log2(n_n0)}, v[{v.v_gtc_in0()}]")

        if c_k1e != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik1e()}], {igemm_log2(n_n0*n_n1b)}, v[{v.v_tmp()}]")
        if c_k0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik0()}], {igemm_log2(n_k1e*n_n0*n_n1b)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_b_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit(f"v_add_u32 v[{v.v_sst_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sst_b_os()}]")
        self._emit_empty_line()

        self._emit(f"; LDS store, wei: k0,k1e,c0,c1: {t_k0}x{t_k1e}x{t_c0}x{t_c1}, {c_k0}x{c_k1e}x{c_c0}x{c_c1}, order:{gemm_m_order}")
        if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1:
            if c_c1 == 1:
                assert c_c0 != 1
                self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c1)}, v[{v.v_gtc_ic0}]")
            else:
                if c_c0 == 1:
                    self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic0()}], {igemm_log2(n_c1)}, v[{v.v_gtc_ic1()}]")
        else:
            if c_c1 == 1:
                assert c_c0 != 1
                self._emit(f"v_mov_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic0}]")
            else:
                if c_c0 == 1:
                    self._emit(f"v_lshlrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c0)}, v[{v.v_gtc_ic1()}]")
                else:
                    self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ic1()}], {igemm_log2(n_c0)}, v[{v.v_gtc_ic0()}]")

        if c_k1e != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik1e()}], {igemm_log2(n_c0*n_c1)}, v[{v.v_tmp()}]")
        if c_k0 != 1:
            self._emit(f"v_lshl_or_b32 v[{v.v_tmp()}], v[{v.v_gtc_ik0()}], {igemm_log2(n_k1e*n_c0*n_c1)}, v[{v.v_tmp()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sst_a_os()}], {igemm_log2(data_byte)}, v[{v.v_tmp()}]")
        self._emit_empty_line()

        self._emit(f"; LDS load")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_b_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_in()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_sld_a_os()}], {igemm_log2(data_byte)}, v[{v.v_gemm_im()}]")
        self._emit(f"v_add_u32 v[{v.v_sld_b_os()}], {self.tunable.lds_a_np2}, v[{v.v_sld_b_os()}]")
        self._emit_empty_line()

        self._emit(self.coalescing_store.init_co_lds_offset(v.v_co_sst(), v.v_co_sld(), v.v_gemm_im(), v.v_gemm_in(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_m_index(v.v_co_sub_m_index(), '0', v.v_tmp()))
        self._emit(self.coalescing_store.init_co_sub_n_index(v.v_co_sub_n_index(), '0', v.v_tmp()))
        self._emit_empty_line()

        self._emit(f"; input offset")
        if gemm_n_unmerge_cluster == 0:
            self._emit(f"s_lshl_b32 s[{s.s_tmp(3)}], s[{s.s_block_gtc_in0()}], {igemm_log2(unmerge_sub_n1 * data_byte)}")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_mul_hi_u32 s[{s.s_tmp(1)}], s[{s.s_in_stride_n()}], s[{s.s_tmp(3)}]")
            self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
            self._emit(f"s_addc_u32 s[{s.s_p_in(1)}], s[{s.s_p_in(1)}], s[{s.s_tmp(1)}]")
        else:
            pass
        self._emit_empty_line()
        self._emit(f"s_lshl_b32 s[{s.s_tmp()}+3], s[{s.s_block_gtc_ic()}], {igemm_log2(data_byte)}")
        self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_in_stride_c()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_mul_hi_u32 s[{s.s_tmp()}+1], s[{s.s_in_stride_c()}], s[{s.s_tmp()}+3]")
        self._emit(f"s_add_u32 s[{s.s_p_in()}], s[{s.s_p_in()}], s[{s.s_tmp()}]")
        self._emit(f"s_addc_u32 s[{s.s_p_in()}+1], s[{s.s_p_in()}+1], s[{s.s_tmp()}+1]")
        self._emit_empty_line()
        self._emit(f"; compute v_co_sub_n_index along n0 x n1b : {n_n0}x{n_n1b}")
        if gemm_n_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_N_N0_N1B:
            if n_n1b != 1:
                self._emit(f"v_and_b32 v[{v.v_in_in1b()}], {n_n1b - 1}, v[{v.v_co_sub_n_index()}]     ; => N1B")
                if n_n0 != 1:
                    self._emit(f"v_lshrrev_b32 v[{v.v_in_in0()}], {igemm_log2(n_n1b)}, v[{v.v_co_sub_n_index()}]  ; => N0")
            else:
                assert n_n0 == self.tunable.block_size
                assert False, "un implemented, should rarely be used"
        else:
            if n_n0 != 1:
                self._emit(f"v_and_b32 v[{v.v_in_in0()}], {n_n0 - 1}, v[{v.v_co_sub_n_index()}]     ; => N0")
                if n_n1b != 0:
                    self._emit(f"v_lshrrev_b32 v[{v.v_in_in1b()}], {igemm_log2(n_n0)}, v[{v.v_co_sub_n_index()}]   ; => N1B")
                else:
                    assert False, "un implemented, should rarely be used"
            else:
                if n_n1b != 0:
                    self._emit(f"v_mov_b32 v[{v.v_in_in1b()}], v[{v.v_co_sub_n_index()}]   ; => N1B")
                else:
                    assert False, "un implemented, should rarely be used"

        self._emit(f";   compute from n1b")
        if self.tunable.nxe != 0:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_in_in1b()}]")
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_in_in1(), v.v_tmp(5), s.s_stride_dslice_hw(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_in_dslice_iw(), v.v_in_dslice_ih(), v.v_tmp(4), s.s_dslice_w(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()
            self._emit(f"v_add_u32 v[{v.v_in_dslice_ih()}], s[{s.s_dslice_h_left()}], v[{v.v_in_dslice_ih()}]")
            self._emit(f"v_add_u32 v[{v.v_in_dslice_iw()}], s[{s.s_dslice_w_left()}], v[{v.v_in_dslice_iw()}]")
            self._emit_empty_line()
            self._emit(f"; dslice_h,dslice_y -> hip,  dslice_w,dslicw_x -> wip")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}], s[{s.s_dtile_iy()}], s[{s.s_dilation_h()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_stride_h()}], v[{v.v_in_dslice_ih()}]")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], s[{s.s_tmp()}], v[{v.v_tmp()}]")
            self._emit(f"s_mul_i32 s[{s.s_tmp()}+1], s[{s.s_dtile_ix()}], s[{s.s_dilation_w()}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}+1], s[{s.s_stride_w()}], v[{v.v_in_dslice_iw()}]")
            self._emit(f"v_add_u32 v[{v.v_tmp()}+1], s[{s.s_tmp()}+1], v[{v.v_tmp()}+1]")
            self._emit(f"; v_tmp: hip, v_tmp+1: wip")
            self._emit_empty_line()
            self._emit(f"; hip->h, wip->w")
            self._emit(f"v_sub_i32 v[{v.v_in_ihi()}], v[{v.v_tmp()}], s[{s.s_pad_h()}]")
            self._emit(f"v_sub_i32 v[{v.v_in_iwi()}], v[{v.v_tmp()}+1], s[{s.s_pad_w()}]")
            self._emit_empty_line()
            self._emit(m_set_flag_hw(v.v_in_flag(), v.v_in_ihi(), v.v_in_iwi(), s.s_hi(), s.s_wi()))
        else:
            self._emit(f"v_add_u32 v[{v.v_tmp(5)}], s[{s.s_block_gtc_in1b()}], v[{v.v_in_in1b()}]")
            self._emit(m_int_div_rem_vs(v.v_tmp(4), v.v_in_in1(), v.v_tmp(5), s.s_stride_hw(), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_vs(v.v_in_iwi(), v.v_in_ihi(), v.v_tmp(4), s.s_wi(), v.v_tmp(), s.s_tmp()))
            self._emit_empty_line()

        self._emit_empty_line()
        self._emit(f"; add in_in0, in_in1")
        if n_n0 != 1:
            if gemm_n_unmerge_cluster == 0:
                self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_in_in0()}], {igemm_log2(unmerge_sub_n1)}, v[{v.v_in_in1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_in_os()}], s[{s.s_in_stride_n()}], v[{v.v_tmp(1)}]")
            else:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_n()}], v[{v.v_in_in1()}]")
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_n0()}], v[{v.v_in_in0()}]")
                self._emit(f"v_add_u32 v[{v.v_in_os()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}]")
        else:
            self._emit(f"v_mul_lo_u32 v[{v.v_in_os()}], s[{s.s_in_stride_n()}], v[{v.v_in_in1()}]")

        self._emit(f"; add i_c")
        if gemm_m_unmerge_cluster == 0:
            if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1:
                self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c() if self.tunable.nxe != 0 else s.s_stride_hw()}], v[{v.v_co_sub_m_index()}]")
            else:
                if n_c0 == 1:
                    self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c() if self.tunable.nxe != 0 else s.s_stride_hw()}], v[{v.v_co_sub_m_index()}]")
                else:
                    if n_c1 == 1:
                        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c() if self.tunable.nxe != 0 else s.s_stride_hw()}], v[{v.v_co_sub_m_index()}]")
                    else:
                        self._emit(f"v_and_b32 v[{v.v_tmp()}], {n_c0 - 1}, v[{v.v_co_sub_m_index()}]        ; => c0")
                        self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(n_c0)}, v[{v.v_co_sub_m_index()}]       ; => c1")
                        self._emit(f"v_lshl_or_b32 v[{v.v_tmp(1)}], v[{v.v_tmp()}], {igemm_log2(n_c1)}, v[{v.v_tmp(1)}]")
                        self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c() if self.tunable.nxe != 0 else s.s_stride_hw()}], v[{v.v_tmp(1)}]")
        else:
            if gemm_m_order == IGEMM_BWD_GTC_LDS_STORE_ORDER_GEMM_M_C0_C1:
                self._emit(f"v_and_b32 v[{v.v_tmp()}], {n_c1 - 1}, v[{v.v_co_sub_m_index()}]    ; => c1")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp(1)}], {igemm_log2(n_c1)}, v[{v.v_co_sub_m_index()}]   ; => c0")
            else:
                self._emit(f"v_and_b32 v[{v.v_tmp(1)}], {n_c0 - 1}, v[{v.v_co_sub_m_index()}]    ; => c0")
                self._emit(f"v_lshrrev_b32 v[{v.v_tmp()}], {igemm_log2(n_c0)}, v[{v.v_co_sub_m_index()}]   ; => c1")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_in_stride_c0()}] ,v[{v.v_tmp(1)}]")
            self._emit(f"v_mul_lo_u32 v[{v.v_tmp()}], s[{s.s_in_stride_c() if self.tunable.nxe != 0 else s.s_stride_hw()}] ,v[{v.v_tmp()}]")
            self._emit(f"v_add_u32 v[{v.v_tmp()}], v[{v.v_tmp()}], v[{v.v_tmp(1)}]")


        self._emit(f"v_add_u32 v[{v.v_in_os()}], v[{v.v_in_os()}], v[{v.v_tmp()}]")
        self._emit(f"; add hi, wi")
        self._emit(f"v_mul_lo_u32 v[{v.v_tmp(1)}], s[{s.s_wi()}], v[{v.v_in_ihi()}]")
        self._emit(f"v_add3_u32 v[{v.v_in_os()}], v[{v.v_in_os()}], v[{v.v_tmp(1)}], v[{v.v_in_iwi()}]")
        self._emit(f"v_lshlrev_b32 v[{v.v_in_os()}], {igemm_log2(data_byte)}, v[{v.v_in_os()}]")

        self._emit(f"; move slice stride")
        assert n_k0 * n_k1e == self.tunable.gemm_k_per_block
        #if n_k0 != 1:
        #    self._emit(f"s_mov_b32 s[{s.s_move_slice_k_k0}], {n_k0}")
        if self.tunable.nxe != 0:
            self._emit(f"s_mov_b32 s[0], {n_k1e}")
            self._emit(m_int_div_rem_ss(s.s_tmp(4), s.s_move_slice_k_k1(), '0', s.s_stride_dslice_yx(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
            self._emit(m_int_div_rem_ss(s.s_move_slice_k_dsx(), s.s_move_slice_k_dsy(), s.s_tmp(4), s.s_dslice_x(), v.v_tmp(4), v.v_tmp(), s.s_tmp()))
        else:
            self._emit(f"s_mov_b32 s[{s.s_move_slice_k_k1()}], {n_k1e}")
        self._emit_empty_line()

        m_move_slice_window = self.get_macro_move_slice_window()


        if self.tunable.nxe != 0:
            assert s.s_out_stride_k.label not in self.dict_shifted_stride and s.s_wei_stride_k.label not in self.dict_shifted_stride
            self._emit(m_move_slice_window.init_stride_k(s.s_out_stride_k(), s.s_wei_stride_k(), s.s_out_stride_k_k1(), s.s_wei_stride_k_k1(),
                                                        s.s_out_stride_k_k0_k1_diff(), s.s_wei_stride_k_k0_k1_diff(), s.s_move_slice_k_k1()))
        else:
            if self.is_1d_move_slice_k():
                self._emit(m_move_slice_window.init_stride_k(s.s_stride_hw(), s.s_c(), s.s_out_stride_k_k1(), s.s_wei_stride_k_k1(), s.s_move_slice_k_k1()))
            else:
                self._emit(m_move_slice_window.init_stride_k(s.s_stride_hw(), s.s_c(), s.s_out_stride_k_k1(), s.s_wei_stride_k_k1(),
                                                        s.s_out_stride_k_k0_k1_diff(), s.s_wei_stride_k_k0_k1_diff(), s.s_move_slice_k_k1()))

        self._emit(self.try_shift_stride(s.s_out_stride_k_k1, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_wei_stride_k_k1, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_out_stride_k_k0_k1_diff, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_wei_stride_k_k0_k1_diff, igemm_log2(data_byte)))

        self._emit(self.try_shift_stride(s.s_out_stride_k, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_wei_stride_k, igemm_log2(data_byte)))
        self._emit(self.try_shift_stride(s.s_in_stride_c, igemm_log2(data_byte)))
        if gemm_m_unmerge_cluster == 1:
            self._emit(self.try_shift_stride(s.s_in_stride_c0, igemm_log2(data_byte)))

        if not self.is_1d_move_slice_k():
            self._emit(f"s_mov_b32 s[{s.s_gemm_k_num_k1()}], {unmerge_sub_k1}")
        if self.tunable.nxe != 0:
            self._emit(f"s_mul_i32 s[{s.s_knum()}], s[{s.s_stride_dslice_yx()}], s[{s.s_k()}]")
        else:
            self._emit(f"s_mov_b32 s[{s.s_knum()}], s[{s.s_k()}]")
        self._emit_empty_line()

    def emit_kernel_fma_main_loop(self):
        s = self.sgpr
        v = self.vgpr

        def move_slice_window_b():
            if self.tunable.nxe != 0:
                m_move_slice_window   = self.get_macro_move_slice_window()
                m_out_update_os       = self.get_macro_out_update_os()
                m_out_update_hw       = self.get_macro_out_update_hw()
                m_set_flag_hw         = self.get_macro_set_flag_hw()
                with self._deferred_context():
                    self._emit(m_move_slice_window(v.v_move_slice_k_ik1(), v.v_move_slice_k_idsy(), v.v_move_slice_k_idsx(), s.s_gemm_k_num_k1(), s.s_gemm_k_num_dsy(), s.s_gemm_k_num_dsx(),
                            s.s_move_slice_k_k1(), s.s_move_slice_k_dsy(), s.s_move_slice_k_dsx(), v.v_out_os_base(), v.v_wei_os_base(),
                            s.s_out_stride_k(), s.s_wei_stride_k(), s.s_out_stride_k_k1(), s.s_wei_stride_k_k1(), s.s_out_stride_k_k0_k1_diff(), s.s_wei_stride_k_k0_k1_diff()))
                    self._emit(m_out_update_hw(v.v_out_iho(), v.v_out_iwo(), v.v_out_dslice_ih(), v.v_out_dslice_iw(), v.v_move_slice_k_idsy(), v.v_move_slice_k_idsx(), s.s_dtile_dy_neg(), s.s_dtile_dx_neg()))
                    self._emit(m_out_update_os(v.v_out_os(), v.v_out_os_base(), v.v_out_iho(), v.v_out_iwo(), s.s_wo(), v.v_tmp()))
                    self._emit(m_set_flag_hw(v.v_out_flag(), v.v_out_iho(), v.v_out_iwo(), s.s_ho(), s.s_wo()))
                return self._get_deferred()
            else:
                m_move_slice_window   = self.get_macro_move_slice_window()
                with self._deferred_context():
                    if self.is_1d_move_slice_k():
                        self._emit(m_move_slice_window(v.v_out_os(), v.v_wei_os(), s.s_out_stride_k_k1(), s.s_wei_stride_k_k1()))
                    else:
                        self._emit(m_move_slice_window(v.v_move_slice_k_ik1(), s.s_gemm_k_num_k1(), 
                                s.s_move_slice_k_k1(), v.v_out_os(), v.v_wei_os(),
                                s.s_out_stride_k_k1(), s.s_wei_stride_k_k1(), s.s_out_stride_k_k0_k1_diff(), s.s_wei_stride_k_k0_k1_diff()))
                return self._get_deferred()


        def move_slice_window_a():
            if self.tunable.nxe != 0:
                m_wei_update_os   = self.get_macro_wei_update_os()
                m_wei_update_yx   = self.get_macro_wei_update_yx()
                with self._deferred_context():
                    self._emit(m_wei_update_yx(v.v_wei_iy(), v.v_wei_ix(), v.v_move_slice_k_idsy(), v.v_move_slice_k_idsx(), s.s_dtile_y(), s.s_dtile_x(), v.v_dtile_iy(), v.v_dtile_ix()))
                    self._emit(m_wei_update_os(v.v_wei_os(), v.v_wei_os_base(), v.v_wei_iy(), v.v_wei_ix(), s.s_x(), v.v_tmp()))
                return self._get_deferred()
            else:
                with self._deferred_context():
                    # we don't really need do anything for a, in nxe 0 case.
                    pass
                return self._get_deferred()

        fctrl                             = ctrl_fma_main_loop_t()
        fctrl.thread_m                    = self.tunable.thread_tile_m
        fctrl.thread_n                    = self.tunable.thread_tile_n
        fctrl.unroll_k                    = self.tunable.gemm_k_per_block
        fctrl.label_prefix                = self.name()
        fctrl.gemm_m_repeat               = self.tunable.gemm_m_repeat
        fctrl.gemm_m_level0_cluster       = self.tunable.gemm_m_level0_cluster
        fctrl.gemm_m_level1_cluster       = self.tunable.gemm_m_level1_cluster
        fctrl.gemm_n_repeat               = self.tunable.gemm_n_repeat
        fctrl.gemm_n_level0_cluster       = self.tunable.gemm_n_level0_cluster
        fctrl.gemm_n_level1_cluster       = self.tunable.gemm_n_level1_cluster
        fctrl.lds_single_size             = self.tunable.lds_single            # in byte, should be power of 2

        # functor
        fctrl.global_load_a_functor       = self.global_load_wei
        fctrl.global_load_b_functor       = self.global_load_out
        fctrl.shared_store_a_functor      = self.shared_store_wei
        fctrl.shared_store_b_functor      = self.shared_store_out
        fctrl.shared_load_a_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_m * 4)
        fctrl.shared_load_b_functor       = inst_ds_read_t(self.tunable.thread_sub_tile_n * 4)
        fctrl.move_slice_window_a_functor = move_slice_window_a
        fctrl.move_slice_window_b_functor = move_slice_window_b

        # sympol type
        fctrl.v_a                         = v.v_a
        fctrl.v_b                         = v.v_b
        fctrl.v_c                         = v.v_c
        fctrl.v_gld_a                     = v.v_gld_a
        fctrl.v_gld_b                     = v.v_gld_b
        fctrl.v_sld_a_os                  = v.v_sld_a_os
        fctrl.v_sld_b_os                  = v.v_sld_b_os
        fctrl.v_sst_a_os                  = v.v_sst_a_os
        fctrl.v_sst_b_os                  = v.v_sst_b_os
        fctrl.s_kitr                      = s.s_kitr
        fctrl.s_knum                      = s.s_knum

        fma_main_loop = fma_main_loop_t(self.mc, fctrl)
        fma_main_loop.emit()

    def emit_kernel_epilogue(self):
        s = self.sgpr
        v = self.vgpr
        #label_out = f"L_{self.name()}_out"

        if self.tunable.nxe != 0:
            self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
                s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_in_stride_c(), s.s_tmp(), v.v_in_flag()))
        else:
            self._emit(self.coalescing_store(v.v_c(), v.v_co_sst(), v.v_co_sld(), s.s_p_in(), v.v_in_os(), None,
                s.s_in_stride_c0() if self.tunable.gemm_m_unmerge_cluster == 1 else None, s.s_in_stride_c(), s.s_tmp()))

        self._emit_front(f"{self.label_out}:")

    def emit_kernel_symbol(self):
        self.karg.emit()
        self._emit_empty_line()
        self.sgpr.emit()
        self._emit_empty_line()
        self.vgpr.emit()
        self._emit_empty_line()

    def emit_kernel_header(self):
        kernel_name = self.name()
        self._emit('.text')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.globl {}'.format(kernel_name))
        self._emit('.p2align 8')
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V3:
            self._emit('.type {},@function'.format(kernel_name))
        if self.mc.arch_config.code_object == AMDGPU_CODEOBJECT_V2:
            self._emit('.amdgpu_hsa_kernel {}'.format(kernel_name))
        self._emit('{}:'.format(kernel_name))

    def emit_kernel_body(self):
        self.emit_kernel_prologue()
        self.emit_kernel_fma_main_loop()
        self.emit_kernel_epilogue()
    def emit_kernel_end(self):
        self._emit('s_endpgm')
    def emit_kernel_footer(self):
        self._emit_empty_line()

    def emit_kernel_amd_kernel_code_t(self):
        amd_kernel_code_t(self.mc, self.get_kernel_info()).emit()
