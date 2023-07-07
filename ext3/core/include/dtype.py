from ext3.typing import *
from .ttype      import Ttype

import torch, re
import qtorch3, qtorch3.quant # type: ignore
import numpy as np

__all__ = [ 'Dtype', 'DtypeRndModl', 'FP32', 'BF16', 'FP16', 'INT' ] # 'FP_6_9_0', 'FP_4_3_4', 'FP_5_2_0' ]

#=======#
# Dtype # <--- type of data (in a tensor) = FP32 | BF16 | ...
#=======#
class Dtype():
    #-----#
    # obj #
    #-----#
    #
    # For e \in {0,1}^exp and f \in {0,1}^man,
    #
    #   [[ (e,f) ]]_(exp, man, exp_bias)
    #   = [[ (e,f) ]]_(exp, man, 0) * 2^(-exp_bias) 
    #   = 2^(e-(2^(exp-1)-1)) * 1.f * 2^(-exp_bias).
    #

    # core.
    fmt      : qtorch3.FloatingPoint
    exp_bias : Opt[int] # None ==> dynamic exp_bias.
    
    # aux.
    maxval_lg   : float # aux. (max_val of fmt) = 2^(2^(fmt.exp-1)) * (2-2^(-fmt.man)).
    cur_exp_bias: Opt[int]   # aux. exp_bias of self that was used in rounding most recently.
    undfl_thrs: float # aux. for exp_bias=0.
    ovrfl_thrs: float # aux. for exp_bias=0.

    def __init__(self, exp: int, man: int, exp_bias: Opt[int]=0) -> None:
        # core.
        self.fmt = qtorch3.FloatingPoint(exp=exp, man=man)
        self.exp_bias = exp_bias
        # aux.
        self.maxval_lg = (2**(exp-1)) #+ np.log2(2-2**(-man))
        self.cur_exp_bias = exp_bias
        self.undfl_thrs = self.get_underflow_thrs(0)
        self.ovrfl_thrs = self.get_overflow_thrs(0)

    # clone Dtype obj.
    def clone(self) -> 'Dtype':
        # NOTE: cur_exp_bias is inited for the cloned obj (important if exp_bias=None).
        return Dtype(self.fmt.exp, self.fmt.man, self.exp_bias)

    # to use Dtype as a key in dict.
    def __hash__(self) -> int:
        return hash((self.fmt.exp, self.fmt.man, self.exp_bias))
        
    # ==, <.
    def __eq__(self, other) -> bool:
        if isinstance(other, Dtype):
            return (self.fmt.exp  == other.fmt.exp and
                    self.fmt.man  == other.fmt.man and
                    self.exp_bias == other.exp_bias)
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, Dtype):
            return (self.fmt.exp < other.fmt.exp and
                    self.fmt.man < other.fmt.man)
        raise ValueError

    # numbit.
    def get_numbit(self) -> int:
        return 1 + self.fmt.exp + self.fmt.man

    # round modl.
    def get_rndmd(self) -> 'DtypeRndModl':
        # NOTE: this func ignores self.exp_bias.
        # round to self.mt in fwd pass; nop in bwd pass (so all infs are kept).
        #-----
        # rndmd = qtorch3.quant.Quantizer(forward_number  = self.fmt, forward_rounding  = 'nearest',
        #                                backward_number = None,     backward_rounding = 'nearest')
        #-----
        rndmd = qtorch3.quant.QuantizerCustom(fwd_num=self.fmt)
        rndmd.dtype = self # type: ignore
        return rndmd # type: ignore
    
    def get_underflow_thrs(self, exp_bias: int) -> float:
        # NOTE: this func ignores self.exp_bias.
        # return val such that in dtype=(self, exp_bias),
        # - x in [0  , val] ==> x rounds to zero.
        # - x in [val, inf] ==> x rounds to non-zero.
        exp, man = self.fmt.exp, self.fmt.man
        return 2**(-2**(exp-1)+2 - exp_bias) * (2**(-man-1))

    def get_overflow_thrs(self, exp_bias: int) -> float:
        # NOTE: this func ignores self.exp_bias.
        # return val such that in dtype=(self, exp_bias),
        # - x in [val, inf] ==> x rounds to inf.
        # - x in [0  , val] ==> x rounds to non-inf.
        exp, man = self.fmt.exp, self.fmt.man
        if self.fmt.exp >= 8:
            exp = 8
            return 2**(2**(exp-1)-1 - exp_bias) * (2-2**(-man-1))
        else:
            return 2**(2**(exp-1)   - exp_bias) * (2-2**(-man-1))

    # Dtype -> str.
    def __repr__(self) -> str:
        # special.
        if self == FP32: return 'FP32'
        if self == BF16: return 'BF16'
        if self == FP16: return 'FP16'
        if self == INT : return 'INT'

        # general.
        if   self.exp_bias is None: exp_bias_str = f'd'
        elif self.exp_bias >= 0:    exp_bias_str = f'{self.exp_bias}'
        else:                       exp_bias_str = f'n{abs(self.exp_bias)}'

        if 0 <= self.fmt.exp <= 9 and 0 <= self.fmt.man <= 9 and len(exp_bias_str) == 1:
            return f'FP_{self.fmt.exp}{self.fmt.man}{exp_bias_str}'
        else:
            return f'FP_{self.fmt.exp}_{self.fmt.man}_{exp_bias_str}'

    #-------#
    # class #
    #-------#
    # str -> Dtype.
    @staticmethod
    def from_str(v: str) -> 'Dtype':
        # special.
        if v == 'FP32': return FP32
        if v == 'BF16': return BF16
        if v == 'FP16': return FP16
        if v == 'INT' : return INT
        # general.
        m = re.compile('FP_(\d+)_(\d+)_(d|(n|)(\d+))$').match(v)
        if m is not None:
            g = m.groups()
            exp, man = int(g[0]), int(g[1])
            if g[2] == 'd': exp_bias = None
            else:           exp_bias = (-1 if g[3] == 'n' else 1) * int(g[4])
            return Dtype(exp, man, exp_bias)
        raise ValueError

    # helper for round_*.
    def _get_exp_bias(self, t: TS, reset_cur_exp_bias: bool) -> int:
        # set: exp_bias.
        exp_bias: int
        if   (reset_cur_exp_bias is True  and self.exp_bias     is not None):
            exp_bias = self.exp_bias
        elif (reset_cur_exp_bias is False and self.cur_exp_bias is not None):
            exp_bias = self.cur_exp_bias
        else:
            # reset_cur_exp_bias=True  ==> set *everytime* if dtype uses dynamic exp_bias.
            # reset_cur_exp_bias=False ==> set *only* if dtype's dynamic exp_bias is not yet set.
            t_abs_max = max( t.abs().max().item(), np.ldexp(1, -100) )
            exp_bias = int(np.floor( self.maxval_lg - np.log2(t_abs_max) ))
            # print(t.abs().max().item(), t_abs_max, self.maxval_lg, exp_bias)
            
        # set: self.cur_exp_bias.
        self.cur_exp_bias = exp_bias
        return exp_bias

    # round with dtype.
    @staticmethod
    def round_dtype(t: Opt[TS], dtype: 'Dtype', allow_inf: bool, emodl, ttype, i,
                    reset_cur_exp_bias: bool=True) -> Opt[TS]:
        # nop if t = None, non-fp data, INT, or FP32.
        if (t is None) or (not t.is_floating_point()) or (dtype == INT) or (dtype == FP32):
            return t
        # assert(isinstance(t, TS) and isinstance(dtype, Dtype) and isinstance(allow_inf, bool))

        # set: t, exp_bias, dtype.cur_exp_bias.
        #----- NOTE: slower.
        # t = t.contiguous()
        #-----
        exp_bias = dtype._get_exp_bias(t, reset_cur_exp_bias)

        # calc: {under,over}flow ratio.
        #----- NOTE: slower.
        # t_abs = t.abs()
        # undfl_count = ((t != 0) * (t_abs <= dtype.get_underflow_thrs(exp_bias))) #.sum() #.count_nonzero()
        # ovrfl_count = (            t_abs >= dtype.get_overflow_thrs (exp_bias) ) #.sum() #.count_nonzero()
        # undfl_ratio = undfl_count / t.numel()
        # ovrfl_ratio = ovrfl_count / t.numel()
        #-----
        # undfl_ratio = qtorch3.quant.ratio_abs_leq_thrs(t, thrs=dtype.undfl_thrs * (2**-exp_bias))
        # ovrfl_ratio = qtorch3.quant.ratio_abs_geq_thrs(t, thrs=dtype.ovrfl_thrs * (2**-exp_bias))
        undovr_ratio = qtorch3.quant.ratio_abs_leq_geq_thrs(t, 
                                                            thrs_l=dtype.undfl_thrs * (2**-exp_bias),
                                                            thrs_g=dtype.ovrfl_thrs * (2**-exp_bias),)
        emodl.info_ts[ttype].undovr[i] = undovr_ratio

        # if ttype == Ttype.GP:
        #     thrs_l = dtype.undfl_thrs * (2**-exp_bias)
        #     print(f'thrs_l={thrs_l}, exp_bias={exp_bias}')
        #     print(f'undrto= {((t.abs() <= thrs_l).sum() - (t==0.).sum()) / t.numel()}')
        #     print(f'{str(type(emodl)):40}, {ttype}, {i}, {dtype}, {undovr_ratio[0]}')

        # round: t.
        # NOTE: handle exp_bias by rnd(t * 2^exp_bias) / 2^exp_bias.
        # NOTE: torch.mul(2**exp) seems faster than torch.ldexp(torch.tensor(exp)).
        #----- NOTE: old ver.
        # res = t
        # if exp_bias != 0:
        #     res = res.mul(np.ldexp(1, +exp_bias))
        #     # res = res.ldexp(torch.tensor(+exp_bias))
        # if allow_inf is False:
        #     # round to +-(max finite val) if overflow occurs.
        #     res = qtorch3.quant.float_quantize    (res, exp=dtype.fmt.exp, man=dtype.fmt.man, rounding='nearest')
        # else:
        #     # round to +-(inf) if overflow occurs.
        #     res = qtorch3.quant.float_quantize_inf(res, exp=dtype.fmt.exp, man=dtype.fmt.man, rounding='nearest')
        # if exp_bias != 0:
        #     res = res.mul(np.ldexp(1, -exp_bias))
        #     # res = res.ldexp(torch.tensor(-exp_bias))
        #-----
        res = qtorch3.quant.float_quantize_custom(t, exp=dtype.fmt.exp, man=dtype.fmt.man,
                                                  exp_bias_pow=2**exp_bias, allow_inf=allow_inf)
        return res

    # round with rndmd.
    @staticmethod
    def round_rndmd(t: Opt[TS], rndmd: 'DtypeRndModl', allow_inf: bool, emodl, ttype, i,
                    reset_cur_exp_bias: bool=True) -> Opt[TS]:
        # nop if t = None, non-fp data, INT, or FP32.
        if (t is None) or (not t.is_floating_point()) or (rndmd.dtype == INT) or (rndmd.dtype == FP32):
            return t
        # assert(isinstance(t, TS) and isinstance(rndmd.dtype, Dtype) and isinstance(allow_inf, bool))

        # set: t, exp_bias, rndmd.dtype.cur_exp_bias.
        #----- NOTE: slower.
        # t = t.contiguous()
        #-----
        exp_bias = rndmd.dtype._get_exp_bias(t, reset_cur_exp_bias)

        # calc: {under,over}flow ratio.
        #----- NOTE: slower.
        # t_abs = t.abs()
        # undfl_count = ((t != 0) * (t_abs <= dtype.get_underflow_thrs(exp_bias))) #.sum() #.count_nonzero()
        # ovrfl_count = (            t_abs >= dtype.get_overflow_thrs (exp_bias) ) #.sum() #.count_nonzero()
        # undfl_ratio = undfl_count / t.numel()
        # ovrfl_ratio = ovrfl_count / t.numel()
        #-----
        # undfl_ratio = qtorch3.quant.ratio_abs_leq_thrs(t, thrs=rndmd.dtype.undfl_thrs * (2**-exp_bias))
        # ovrfl_ratio = qtorch3.quant.ratio_abs_geq_thrs(t, thrs=rndmd.dtype.ovrfl_thrs * (2**-exp_bias))
        undovr_ratio = qtorch3.quant.ratio_abs_leq_geq_thrs(t, 
                                                            thrs_l=rndmd.dtype.undfl_thrs * (2**-exp_bias),
                                                            thrs_g=rndmd.dtype.ovrfl_thrs * (2**-exp_bias),)
        emodl.info_ts[ttype].undovr[i] = undovr_ratio

        # round: t.
        # NOTE: handle exp_bias by rnd(t * 2^exp_bias) / 2^exp_bias.
        # NOTE: torch.mul(2**exp) seems faster than torch.ldexp(torch.tensor(exp)).
        #----- NOTE: old ver.
        # res = t
        # if exp_bias != 0:
        #     res = res.mul(np.ldexp(1, +exp_bias))
        #     # res = res.ldexp(torch.tensor(+exp_bias))
        # res = rndmd(res)
        # if exp_bias != 0:
        #     res = res.mul(np.ldexp(1, -exp_bias))
        #     # res = res.ldexp(torch.tensor(-exp_bias))
        #-----
        res = rndmd(t, exp_bias_pow=2**exp_bias, allow_inf=allow_inf)
        return res

#==============#
# DtypeRndModl #
#==============#
class DtypeRndModl(torch.nn.Module):
    dtype: Dtype

#=======#
# const #
#=======#
FP32 = Dtype(8, 23, 0)
BF16 = Dtype(8,  7, 0)
FP16 = Dtype(5, 10, 0)
INT  = Dtype(1,  1, 0)
# FP_6_9_0 = Dtype(6,  9, 0)
# FP_4_3_4 = Dtype(4,  3, 4)
# FP_5_2_0 = Dtype(5,  2, 0)
