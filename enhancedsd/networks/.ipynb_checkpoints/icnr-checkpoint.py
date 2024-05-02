"""
This file requires no change
"""
import torch
import torch.nn as nn

def ICNR(tensor, initializer, upscale_factor=2, *args, **kwargs):
    """
    ICNR Initialization
    
    A method to void checkerboard artifacts, refer to the following papers: 
            - Checkerboard artifact free sub-pixel convolution: A note on sub-pixel convolution, resize convolution and convolution resize Arxiv 2017
            - EnhancedSD: Downscaling Solar Irradiance from Climate Model Projections, CCAI workshop NeurIPS 22
            
    + tensor: the 2-dimensional Tensor or more
    """
    
    upscale_factor_squared = upscale_factor * upscale_factor
    assert tensor.shape[0] % upscale_factor_squared == 0, \
        ("The size of the first dimension: "
         f"tensor.shape[0] = {tensor.shape[0]}"
         " is not divisible by square of upscale_factor: "
         f"upscale_factor = {upscale_factor}")
    sub_kernel = torch.empty(tensor.shape[0] // upscale_factor_squared,
                             *tensor.shape[1:])
    sub_kernel = initializer(sub_kernel, *args, **kwargs)
    return sub_kernel.repeat_interleave(upscale_factor_squared, dim=0)