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

import math
from .igemm_base import *
from .shared_memory import *
from .global_memory import *
from .thread_mapping import *
import copy

IGEMM_COALESCING_GEMM_M_ORDER_M0_M1 = 0
IGEMM_COALESCING_GEMM_M_ORDER_M1_M0 = 1

class ctrl_coalescing_store_t(object):
    def __init__(self):
        self.ctm = None # ctrl_thread_mapping_t
        self.coalescing_groups = 1
        self.block_size = 256
        self.vector_store_size = 1
        self.data_byte = 1
        self.gemm_m_order = IGEMM_COALESCING_GEMM_M_ORDER_M0_M1
        self.gemm_m_m0_m1 = []
        # for simplicity, if order is IGEMM_COALESCING_GEMM_M_ORDER_M1_M0, coalescing_groups must be multiply of t_m0
        # use adjust_optimal_coalescing_groups() to do this.

        # self.s_unmerge_cluster_stride_m0 = None     # if not none, means M0*M1 is not continous

    def adjust_optimal_coalescing_groups(self):
        '''
        if coalescing_groups is not multiply of t_m0, we will be very hard to calculate thread stride after load from LDS
        this can help make easy index calculation, performance should be no impact
        '''
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M1_M0:
            cg = self.coalescing_groups
            while cg % self.ctm.t_m0() != 0:
                cg = cg * 2
            assert cg <= self.get_length_m_groups()
            self.coalescing_groups = cg

    def get_length_m_groups(self):
        return self.ctm.t_mr() * self.ctm.t_m1() * self.ctm.t_m0()
    def get_length_n_groups(self):
        return self.ctm.t_nr() * self.ctm.t_n1() * self.ctm.t_n0()

    def get_subgroups(self):
        # num_m_group * num_n_group = num_group
        l_mg = self.get_length_m_groups()
        l_ng = self.get_length_n_groups()
        num_m_groups = math.gcd(self.coalescing_groups, l_mg)
        num_n_groups = math.gcd(self.coalescing_groups//num_m_groups, l_ng)
        g_mr = math.gcd(num_m_groups, self.ctm.t_mr())
        g_m1 = math.gcd(num_m_groups//g_mr, self.ctm.t_m1())
        g_m0 = math.gcd(num_m_groups//(g_mr*g_m1), self.ctm.t_m0())
        g_nr = math.gcd(num_n_groups, self.ctm.t_nr())
        g_n1 = math.gcd(num_n_groups//g_nr, self.ctm.t_n1())
        g_n0 = math.gcd(num_n_groups//(g_nr*g_n1), self.ctm.t_n0())
        return g_mr, g_m1, g_m0, g_nr, g_n1, g_n0   # g_m1 always 1

    def get_transposed_thread_mapping(self):
        # after coalescing, the thread mapping is indeed transposed
        assert self.ctm.t_n0() % self.vector_store_size == 0
        assert self.block_size == (self.ctm.c_mr()*self.ctm.c_nr()*self.ctm.c_m0()*self.ctm.c_n0()*self.ctm.c_m1()*self.ctm.c_n1())
        n_n_total = self.ctm.n_n_total()
        n_m_total = self.ctm.n_m_total()
        assert n_n_total % self.vector_store_size == 0
        #trans_t_mr, trans_t_nr, trans_t_m1, trans_t_m0, trans_t_n1, trans_t_n0 = 1,1,1,1,1,1
        #trans_c_mr, trans_c_nr, trans_c_m1, trans_c_m0, trans_c_n1, trans_c_n0 = 1,1,1,1,1,1

        trans_t_n0 = self.vector_store_size
        trans_c_n0 = n_n_total // trans_t_n0
        trans_t_m0 = 1
        trans_c_m0 = self.block_size // trans_c_n0
        trans_t_n1 = 1
        trans_c_n1 = 1
        trans_t_m1 = n_m_total // (trans_t_m0 * trans_c_m0)
        trans_c_m1 = 1
        trans_t_nr = 1
        trans_c_nr = 1
        trans_t_mr = 1
        trans_c_mr = 1

        transposed_thread_mapping = ctrl_thread_mapping_t()
        transposed_thread_mapping.thread_lengths =  [trans_t_mr, trans_t_nr, trans_t_m1, trans_t_n1, trans_t_m0, trans_t_n0]
        transposed_thread_mapping.cluster_lengths = [trans_c_mr, trans_c_nr, trans_c_m1, trans_c_n1, trans_c_m0, trans_c_n0]

        return transposed_thread_mapping

    # def get_group_m_stride(self):
    #     ttm = self.get_transposed_thread_mapping()
    #     assert ttm.t_m0() == 1
    #     
    #     g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
    # 
    #     group_m_stride_mr = 0
    #     group_m_stride_m1 = 0
    #     group_m_stride_m0 = 0
    #     if g_mr != 1:
    #         group_m_stride_mr = (self.ctm.t_mr() // g_mr) * self.ctm.n_m1()*self.ctm.n_m0()
    #     if g_m1 != 1:
    #         group_m_stride_m1 = (self.ctm.t_m1() // g_m1) * self.ctm.n_m0()
    #     if g_m0 != 1:
    #         group_m_stride_m0 = (self.ctm.t_m0() // g_m0)
    # 
    #     return group_m_stride_mr, group_m_stride_m1, group_m_stride_m0  # group_m_stride_m1 may always 1

    def get_thread_m_stride(self):
        ttm = self.get_transposed_thread_mapping()
        assert ttm.t_m0() == 1
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_m1 == 1

        # do some assert
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            m_index_per_group = self.get_m_index_per_group()
            thread_m_stride = g_m0 * ttm.n_m0()
        else:
            m_index_per_group = self.get_m_index_per_group_m1_m0()
            thread_m_stride = -1
        for ig in range(len(m_index_per_group)):
            for ic in range(len(m_index_per_group[ig])):
                _list = m_index_per_group[ig][ic]
                for idx in range(len(_list) - 1):
                    diff_m = _list[idx+1] - _list[idx]
                    if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
                        assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"
                    else:
                        if thread_m_stride == -1:
                            thread_m_stride = diff_m
                        assert diff_m == thread_m_stride, f"diff_m:{diff_m}, thread_m_stride:{thread_m_stride}"

        return thread_m_stride

    def get_num_dword_per_group(self):
        return (self.ctm.t_mr()*self.ctm.t_nr()*self.ctm.t_m0()*self.ctm.t_n0()*self.ctm.t_m1()*self.ctm.t_n1()) // self.coalescing_groups

    def get_sub_m0_offset(self, i_c_m0):
        ttm = self.get_transposed_thread_mapping()
        assert ttm.t_m0() == 1
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_m1 == 1
        sub_m0_offset = ((i_c_m0 >> int(math.log2(g_m0))) << self.ctm.t_m0()) | (i_c_m0 & (g_m0 - 1))
        return sub_m0_offset
        #print(" i_c_m0 >> igemm_log2(g_m0) << self.t_m0   i_c_m0 & (g_m0 - 1)  ")

    def get_m0_m1_index(self, m_index):
        assert len(self.gemm_m_m0_m1) != 0
        m0, m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        # print(f"m0:{m0}, m1:{m1}")
        assert m_index < m0 * m1, f"m_index:{m_index} larger than gemm_m_m0_m1:{self.gemm_m_m0_m1}, please check"
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index // m1, m_index % m1
        else:
            return m_index % m0, m_index // m0

    def get_m_index_per_group(self):
        num_dword_per_group = self.get_num_dword_per_group()
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.get_subgroups()
        assert g_nr == 1 and g_n1 == 1 and g_n0 == 1
        assert g_m1 == 1
        l_mr = self.ctm.t_mr() // g_mr
        l_m0 = self.ctm.t_m0() // g_m0
        ttm = self.get_transposed_thread_mapping()
        m_index_per_group = list()
        for i_g_mr in range(g_mr):
            for i_g_m0 in range(g_m0):
                m_idx_start_per_group = i_g_mr * self.ctm.n_m0()*self.ctm.n_m1() + i_g_m0 * l_m0
                # first, generate current group, m index
                m_index = []
                for t_mr in range(l_mr):
                    for tid_along_m0 in range(self.ctm.c_m0()):
                        for tid_along_m1 in range(self.ctm.c_m1()):
                            m_idx_start_per_m0 = m_idx_start_per_group + tid_along_m1 * self.ctm.n_m0() + tid_along_m0 * self.ctm.t_m0()
                            for t_m0 in range(l_m0):
                                m_idx_start = m_idx_start_per_m0 + t_mr*self.ctm.n_m0()*self.ctm.n_m1() + t_m0
                                m_index.append(m_idx_start)
                assert len(m_index) == ttm.c_m0() * num_dword_per_group, f"{len(m_index)}, {ttm.c_m0()}, {num_dword_per_group}"
                m_index.sort()
                #print(f"xxxxx:{m_index}")
                pixel_m_index = [None] * ttm.c_m0()
                # second, record index according to current id
                for i, m in enumerate(m_index):
                    _icm0 = i % ttm.c_m0()
                    if not pixel_m_index[_icm0]:
                        pixel_m_index[_icm0] = list()
                    pixel_m_index[_icm0].append(m)
                m_index_per_group.append(pixel_m_index)
        return m_index_per_group

    def get_m_index_from_m1_m0(self, m_idx):
        assert len(self.gemm_m_m0_m1) == 2
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_idx
        n_m0, n_m1 = self.gemm_m_m0_m1[0], self.gemm_m_m0_m1[1]
        i_m0 = m_idx % n_m0
        i_m1 = m_idx // n_m0
        return i_m0 * n_m1 + i_m1

    def get_m_index_per_group_m1_m0(self):
        m_index_per_group = copy.copy(self.get_m_index_per_group())
        if self.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            return m_index_per_group
        assert len(self.gemm_m_m0_m1) == 2
        assert len(m_index_per_group) == self.coalescing_groups
        for ig in range(len(m_index_per_group)):
            for i_cm0 in range(len(m_index_per_group[ig])):
                for i_t in range(len(m_index_per_group[ig][i_cm0])):
                    m_index = m_index_per_group[ig][i_cm0][i_t]
                    m_index_per_group[ig][i_cm0][i_t] = self.get_m_index_from_m1_m0(m_index)
        return m_index_per_group

class igemm_coalescing_store_t(mc_base_t):
    def __init__(self, mc, ctrl):
        mc_base_t.__init__(self, mc)
        assert type(ctrl) is ctrl_coalescing_store_t
        assert ctrl.ctm.t_mr() == 2 and ctrl.ctm.t_nr() == 2
        assert ctrl.block_size % ctrl.ctm.n_n_total() == 0, f"block_size:{ctrl.block_size}, gemm_n:{ctrl.ctm.n_n_total()}"
        self.ctrl = ctrl

    def name(self):
        return ''

    def init_co_lds_offset(self, v_co_sst, v_co_sld, v_gemm_im, v_gemm_in, v_tid, v_tmp2):
        ctrl = self.ctrl
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.ctrl.get_subgroups()
        with self._deferred_context():
            self._emit(f"; init_co_lds_offset")
            gemm_m_shrink = g_m0
            if gemm_m_shrink != 1:
                self._emit(f"v_lshrrev_b32 v[{v_tmp2}], {igemm_log2(gemm_m_shrink)}, v[{v_gemm_im}]")
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_tmp2}], {igemm_log2(ctrl.ctm.n_m_total())}, v[{v_gemm_in}]")
            else:
                self._emit(f"v_lshl_or_b32 v[{v_co_sst}], v[{v_gemm_im}], {igemm_log2(ctrl.ctm.n_m_total())}, v[{v_gemm_in}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sst}], {igemm_log2(ctrl.data_byte)}, v[{v_co_sst}]")
            self._emit(f"v_lshlrev_b32 v[{v_co_sld}], {igemm_log2(ctrl.data_byte * ctrl.vector_write_out)}, v[{v_tid}]")

        return self._get_deferred()


    def init_co_sub_m_index(self, v_co_sub_m_index, v_tid, v_tmp2):
        ctrl = self.ctrl
        # need use v_co_sub_m_index to calculate v offset in m direction
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = ctrl.get_subgroups()
        assert g_m1 == 1
        # sub_m0_offset = ((i_c_m0 >> int(math.log2(g_m0))) << self.ctm.t_m0()) | (i_c_m0 & (g_m0 - 1))
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0

        with self._deferred_context():
            self._emit(f"; init_co_sub_m_index")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.ctm.n_n_total())}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_lshrrev_b32 v[{v_co_sub_m_index}], {igemm_log2(ctrl.ctm.n_n_total())}, v[{v_tmp2}]")

            self._emit(f"v_and_b32 v[{v_tmp2}], {l_m0 - 1}, v[{v_co_sub_m_index}]")
            self._emit(f"v_lshrrev_b32 v[{v_tmp2}+1], {igemm_log2(l_m0)}, v[{v_co_sub_m_index}]")
            self._emit(f"v_lshl_or_b32 v[{v_co_sub_m_index}] v[{v_tmp2}+1], {igemm_log2(ctrl.ctm.t_m0())}, v[{v_tmp2}]")
        return self._get_deferred()

    def init_co_sub_n_index(self, v_co_sub_n_index, v_tid, v_tmp2):
        ctrl = self.ctrl
        # need use v_co_sub_n_index to calculate v offset in n direction
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = ctrl.get_subgroups()
        assert g_m1 == 1
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0

        with self._deferred_context():
            self._emit(f"; init_co_sub_n_index")
            if ctrl.vector_write_out == 1:
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.ctm.n_n_total() - 1}, v[{v_tid}]")
            else:
                self._emit(f"v_lshlrev_b32 v[{v_tmp2}], {igemm_log2(ctrl.vector_write_out)}, v[{v_tid}]")
                self._emit(f"v_and_b32 v[{v_co_sub_n_index}], {ctrl.ctm.n_n_total() - 1}, v[{v_tmp2}]")
        return self._get_deferred()


    def __call__(self, v_c, v_co_sst, v_co_sld, s_p_out, v_out_offset, s_out_offset, s_gemm_m0_stride, s_gemm_m1_stride, s_tmp4, v_store_flag = None):
        # if no need s_out_offset, set to integer 0
        # if no need flag to dicide store, set v_store_flag to 0
        ctrl = self.ctrl
        v_c = sym_t(v_c)
        v_co_sst = sym_t(v_co_sst)
        v_co_sld = sym_t(v_co_sld)
        s_tmp4 = sym_t(s_tmp4)
        g_mr, g_m1, g_m0, g_nr, g_n1, g_n0 = self.ctrl.get_subgroups()
        assert g_m1 == 1 and g_nr == 1 and g_n1 == 1 and g_n0 == 1
        l_mr = ctrl.ctm.t_mr() // g_mr
        l_m0 = ctrl.ctm.t_m0() // g_m0
        no_s_out_offset = s_out_offset is None

        # mc, vec_count, vec_byte, vec_stride, sst_base=0):
        inst_sst = inst_ds_write2_likely_t(self.mc, 2, ctrl.ctm.t_n0() * ctrl.data_byte, ctrl.ctm.n_n_total() * ctrl.data_byte // 2)
        # mc, vec_count, vec_byte, vec_stride, sld_base = 0):
        inst_sld = inst_ds_read2_likely_t(self.mc, 2, ctrl.vector_write_out * ctrl.data_byte, ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte)
        # self, vdata, vaddr, srsrc, soffset, offset):
        inst_gst = inst_buffer_store_dword_t(ctrl.vector_write_out)

        s_out_offset_itr = sym_t(s_tmp4(0))
        s_thread_m_stride = sym_t(s_tmp4(1))

        if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
            m_index_per_group = ctrl.get_m_index_per_group()
        else:
            m_index_per_group = ctrl.get_m_index_per_group_m1_m0()
        thread_m_stride = ctrl.get_thread_m_stride()

        assert len(m_index_per_group) == ctrl.coalescing_groups


        with self._deferred_context():
            self._emit(f"; coalescing store, gemm_mxn:{ctrl.ctm.n_m_total()}x{ctrl.ctm.n_n_total()}, block:{ctrl.block_size}, m_repeatxm_perthread:{ctrl.ctm.t_mr()}x{ctrl.ctm.t_m0()}, n_repeatxn_perthread:{ctrl.ctm.t_nr()}x{ctrl.ctm.t_n0()}")
            self._emit(f"; coalescing_groups:{ctrl.coalescing_groups}, num_dword_per_group:{ctrl.get_num_dword_per_group()}")
            self._emit(f"; coalescing_groups_in_m_repeat:{g_mr}, group_length_in_m_repeat:{l_mr}, coalescing_groups_in_m_per_thread:{g_m0}, group_length_in_m_per_thread:{l_m0}")
            # emit some pre index
            if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M1_M0 and s_gemm_m0_stride is not None:
                self._emit(f"s_mul_i32 s[{s_thread_m_stride()}], {thread_m_stride}, s[{s_gemm_m0_stride}]    ; init per thread stride in m dimension")
            else:
                self._emit(f"s_mul_i32 s[{s_thread_m_stride()}], {thread_m_stride}, s[{s_gemm_m1_stride}]    ; init per thread stride in m dimension")

            for i_group in range(ctrl.coalescing_groups):
                m_index_start_per_group = m_index_per_group[i_group][0][0]
                m0_index_start_per_group, m1_index_start_per_group = ctrl.get_m0_m1_index(m_index_start_per_group)

                c_group_start_index = i_group * l_mr * l_m0 * ctrl.ctm.t_n0() * ctrl.ctm.t_nr()
                current_m_index = m_index_per_group[i_group]
                self._emit(f"; start group {i_group}, m index start from {m_index_start_per_group}")
                self._emit(f"s_barrier")
                for i_sub_length in range(l_mr * l_m0):
                    c_sub_start_index = c_group_start_index + i_sub_length * ctrl.ctm.t_n0() * ctrl.ctm.t_nr()
                    sst_offset = i_sub_length * ctrl.ctm.n_n_total() * ctrl.data_byte
                    self._emit(inst_sst(v_co_sst(), v_c(c_sub_start_index), sst_offset))

                self._emit(f"s_waitcnt lgkmcnt(0)")
                self._emit(f"s_barrier")
                self._emit(f";   load from lds")
                issue_list = []
                for i_d in range(ctrl.get_num_dword_per_group() // (2 * ctrl.vector_write_out)):
                    c_sub_start_index = c_group_start_index + i_d * 2 * ctrl.vector_write_out
                    sld_offset = i_d * 2 * ctrl.block_size * ctrl.vector_write_out * ctrl.data_byte
                    self._emit(inst_sld(v_c(c_sub_start_index), v_co_sld(), sld_offset))
                    issue_list.append(inst_sld.get_issues(sld_offset))

                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"v_cmpx_eq_u32 vcc, 1, v[{v_store_flag}]")
                    #self._emit(f"s_cbranch_execz {label_prefix}_co_{i_group}")

                self._emit(f";   store to global, m index start from {m_index_start_per_group}, m0:{m0_index_start_per_group}, m1:{m1_index_start_per_group}")
                if s_gemm_m0_stride is not None:
                    self._emit(f"s_mul_i32 s[{s_tmp4(2)}], {m0_index_start_per_group}, s[{s_gemm_m0_stride}]")
                    self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {m1_index_start_per_group}, s[{s_gemm_m1_stride}]")
                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                else:
                    if m_index_start_per_group == 0:
                        if no_s_out_offset:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], 0")
                        else:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_out_offset}]")
                    elif m_index_start_per_group == 1:
                        if no_s_out_offset:
                            self._emit(f"s_mov_b32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}]")
                        else:
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_gemm_m1_stride}], s[{s_out_offset}]")
                    else:
                        if no_s_out_offset:
                            self._emit(f"s_mul_i32 s[{s_out_offset_itr()}], {m_index_start_per_group}, s[{s_gemm_m1_stride}]")
                        else:
                            self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {m_index_start_per_group}, s[{s_gemm_m1_stride}]")
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(3)}], s[{s_out_offset}]")
                i_m0_start, i_m1_start =  m0_index_start_per_group, m1_index_start_per_group
                for i_gst in range(ctrl.get_num_dword_per_group() // ctrl.vector_write_out):
                    if i_gst % 2 == 0:
                        i_issues =  (i_gst // 2) + 1
                        i_issue_list = issue_list[i_issues:]
                        i_issue_cnt = igemm_flatten_list_accumulate(i_issue_list) if len(i_issue_list) != 0 else 0
                        self._emit(f"s_waitcnt lgkmcnt({i_issue_cnt})")
                    # vdata, vaddr, srsrc, soffset, offset
                    self._emit(inst_gst(v_c(c_group_start_index + i_gst*ctrl.vector_write_out), v_out_offset, s_p_out, s_out_offset_itr(), 0))
                    if i_gst != (ctrl.get_num_dword_per_group() // ctrl.vector_write_out) - 1:
                        if s_gemm_m0_stride is not None:
                            i_m = m_index_per_group[i_group][0][i_gst+1]
                            i_m0, i_m1 = ctrl.get_m0_m1_index(i_m)
                            self._emit(f"; im:{i_m}, i_m0:{i_m0}, i_m1:{i_m1}")
                            if ctrl.gemm_m_order == IGEMM_COALESCING_GEMM_M_ORDER_M0_M1:
                                if i_m0 > i_m0_start:
                                    i_m0_start = i_m0
                                    # m0 accumulate
                                    self._emit(f"s_mul_i32 s[{s_tmp4(2)}], {i_m0}, s[{s_gemm_m0_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                            else:
                                if i_m1 > i_m1_start:
                                    i_m1_start = i_m1
                                    # m1 accumllate
                                    self._emit(f"s_mul_i32 s[{s_tmp4(3)}], {i_m1}, s[{s_gemm_m1_stride}]")
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_tmp4(2)}], s[{s_tmp4(3)}]")
                                else:
                                    self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                        else:
                            self._emit(f"s_add_u32 s[{s_out_offset_itr()}], s[{s_thread_m_stride()}], s[{s_out_offset_itr()}]")
                if v_store_flag is not None and type(v_store_flag) is str:
                    self._emit(f"s_mov_b64 exec, -1")
        return self._get_deferred()
