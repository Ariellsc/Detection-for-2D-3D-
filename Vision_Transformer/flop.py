import torch
from fvcore.nn import FlopCountAnalysis



def main():
    """
    fvcore是Facebook开源的一个轻量级的核心库，它提供了各种计算机视觉框架中常见且基本的功能。
    其中就包括了统计模型的参数以及FLOPs等。
    """
    # Self-Attention
