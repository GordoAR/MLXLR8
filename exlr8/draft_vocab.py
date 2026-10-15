"""MXFP4 vocabulary projection for draft proposals; target logits stay FP8."""
import mlx.core as mx
import mlx.nn as nn

from . import vocab_projection

@mx.compile
def project(x,packed,exponents,scales):
    product=mx.quantized_matmul(x.astype(mx.float32),packed,scales=exponents,mode="mxfp4")
    return (product*scales).astype(x.dtype)

@mx.compile
def refined_project(x,packed,exponents,scales,original):
    coarse=project(x,packed,exponents,scales)
    indices=mx.argpartition(-coarse.reshape(-1),127)[:128].astype(mx.uint32)
    precise=vocab_projection.project(x,original[indices],scales[indices])
    return mx.put_along_axis(coarse,indices[None,None,:],precise,axis=-1)

class DraftVocab(nn.Module):
    def __init__(self,head):
        super().__init__()
        assert head.format==5 and head.in_features%32==0
        self.scales=head.scales
        self.original=head.packed
        chunks=[]
        for start in range(0,head.out_features,1024):
            decoded=mx.dequantize(head.packed[start:start+1024],head.exponents[start:start+1024],
                                 mode="mxfp8",dtype=mx.float32)
            pair=mx.quantize(decoded,mode="mxfp4")
            mx.eval(pair)
            chunks.append(pair)
        self.packed,self.exponents=(mx.concatenate([pair[i] for pair in chunks]) for i in range(2))
        mx.eval(self.packed,self.exponents)
        self.freeze()

    def __call__(self,x):
        assert x.shape==(1,1,6144) and x.dtype==mx.bfloat16
        return refined_project(x,self.packed,self.exponents,self.scales,self.original)
