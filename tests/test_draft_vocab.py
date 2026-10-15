"""Check draft-head encoding, independent products, and target isolation."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from exlr8.dense import DenseLinear
from exlr8.draft_vocab import DraftVocab, project, refined_project
from exlr8.mtp import MTPRuntime
from exlr8.model import Model, DecoderLayer

@pytest.mark.parametrize("n",[31,1024,1025])
@pytest.mark.parametrize("k",[128,256])
@pytest.mark.parametrize("dtype",[mx.bfloat16,mx.float32])
def test_chunked_encoding_matches_independent_decoded_product(n,k,dtype):
    mx.random.seed(41)
    packed,_=mx.quantize(mx.random.normal((n,k)),mode="mxfp8")
    scales=mx.linspace(.001,.004,n)
    head=DenseLinear((5,n,k,0,0),{"fp8":packed.view(mx.uint8),"scale":scales})
    before=(np.array(head.packed),np.array(head.exponents),np.array(head.scales))
    draft=DraftVocab(head)
    x=mx.random.normal((1,1,k)).astype(dtype)
    decoded=mx.dequantize(draft.packed,draft.exponents,mode="mxfp4",dtype=mx.float32)
    expected=((np.asarray(x.astype(mx.float32))@np.asarray(decoded).T)*np.asarray(scales))
    expected=np.asarray(mx.array(expected).astype(dtype).astype(mx.float32))
    actual=project(x,draft.packed,draft.exponents,draft.scales)
    assert actual.dtype==dtype and actual.shape==(1,1,n)
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),expected,rtol=.01,atol=2e-4)
    for value,original in zip((head.packed,head.exponents,head.scales),before):
        np.testing.assert_array_equal(np.asarray(value),original)
    assert draft.scales is head.scales
    assert draft.packed.shape==(n,k//8) and draft.exponents.shape==(n,k//32)

@pytest.mark.parametrize("optimize",[False,True])
@pytest.mark.parametrize("supported",[False,True])
def test_constructor_changes_only_supported_draft_head(monkeypatch,optimize,supported):
    from test_model import fixtures,Linear
    args,weights,experts,vocab=fixtures(topk=3)
    weights=dict(weights,**{"eh_proj":Linear(np.zeros((8,16),np.float32)),
        "enorm.weight":mx.ones(8),"hnorm.weight":mx.ones(8),"shared_head.norm.weight":mx.ones(8)})
    head=DenseLinear((5,8,256,0,0),{"fp8":mx.zeros((8,256),mx.uint8),"scale":mx.ones(8)})
    if supported:head.out_features,head.in_features=154880,6144
    vocab["lm_head"]=head
    target=Model(args,[DecoderLayer(args,0,weights)],vocab)
    sentinel=SimpleNamespace(name="draft")
    seen=[]
    def create(value):
        seen.append(value)
        return sentinel
    monkeypatch.setattr("exlr8.mtp.DenseStore",lambda *a,**kw:SimpleNamespace(layer=lambda i:weights))
    monkeypatch.setattr("exlr8.draft_vocab.DraftVocab",create)
    runtime=MTPRuntime(target,"unused",SimpleNamespace(layer=lambda index:experts),optimize=optimize)
    assert target.lm_head is head
    assert runtime.draft_head is (sentinel if optimize and supported else head)
    assert seen==([head] if optimize and supported else [])

@pytest.mark.parametrize("n",[256,2048])
def test_selected_scores_match_original_head_and_other_scores_stay_coarse(n):
    from exlr8 import vocab_projection
    mx.random.seed(42)
    k=6144
    packed,_=mx.quantize(mx.random.normal((n,k)),mode="mxfp8")
    head=DenseLinear((5,n,k,0,0),{"fp8":packed.view(mx.uint8),"scale":mx.linspace(.001,.004,n)})
    draft=DraftVocab(head)
    x=mx.random.normal((1,1,k)).astype(mx.bfloat16)
    coarse=project(x,draft.packed,draft.exponents,draft.scales)
    indices=mx.argpartition(-coarse.reshape(-1),127)[:128].astype(mx.uint32)
    actual=draft(x)
    full=vocab_projection.project(x,head.packed,head.scales)
    np.testing.assert_array_equal(np.asarray(actual[...,indices].astype(mx.float32)),
                                  np.asarray(full[...,indices].astype(mx.float32)))
    chosen=np.asarray(indices);other=np.ones(n,bool);other[chosen]=False
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32))[...,other],
                                  np.asarray(coarse.astype(mx.float32))[...,other])
    decoded=mx.dequantize(head.packed[indices],head.exponents[indices],mode="mxfp8",dtype=mx.float32)
    expected=((x.astype(mx.float32)@decoded.T)*head.scales[indices]).astype(mx.bfloat16)
    np.testing.assert_allclose(np.asarray(actual[...,indices].astype(mx.float32)),
                              np.asarray(expected.astype(mx.float32)),rtol=.02,atol=.02)
    assert draft.original is head.packed


def test_precise_rows_correct_a_wrong_coarse_winner():
    codes=np.zeros((256,256),np.uint8)
    codes[0,0]=0x44 # exact E4M3 value 3
    codes[1,0]=0x38 # exact E4M3 value 1
    original=mx.array(codes).view(mx.uint32)
    approximated=np.zeros((256,256),np.float32)
    approximated[0,0]=1
    approximated[1,0]=2
    packed,exponents=mx.quantize(mx.array(approximated),mode="mxfp4")
    x=np.zeros((1,1,256),np.float32);x[0,0,0]=1
    x=mx.array(x).astype(mx.bfloat16)
    scales=mx.ones(256,mx.float32)
    coarse=project(x,packed,exponents,scales)
    refined=refined_project(x,packed,exponents,scales,original)
    assert int(mx.argmax(coarse).item())==1
    assert int(mx.argmax(refined).item())==0
    assert float(refined[0,0,0].item())==3 and float(refined[0,0,1].item())==1
