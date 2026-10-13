# Attribution

MLXLR8 adds the EXLR8 loader, MLX integration, and tested kernel changes. The underlying model logic, arithmetic, and EXL3 format remain credited to their upstream authors.

| Project | Work used | License and source |
|---|---|---|
| MLX-LM | GLM attention and routing equations; tokenizer, generation, cache, and HTTP server | [MIT](third_party/licenses/mlx-lm-MIT.txt), [repository](https://github.com/ml-explore/mlx-lm), [GLM DSA model](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/models/glm_moe_dsa.py) |
| oMLX | GLM DSA attention and routing logic used to check the local model | [Apache 2.0](third_party/licenses/oMLX-Apache-2.0.txt), [repository](https://github.com/jundot/omlx), [GLM model](https://github.com/jundot/omlx/blob/main/omlx/patches/glm_moe_dsa/glm_moe_dsa_model.py). The inherited GLM model also carries Apple's copyright. |
| MLX | GPU arithmetic, matrix-vector templates, RMS reduction, typed sigmoid, and array execution | [MIT](third_party/licenses/MLX-MIT.txt), [repository](https://github.com/ml-explore/mlx), [0.32.3 Metal sources](https://github.com/ml-explore/mlx/tree/v0.32.3/mlx/backend/metal/kernels) |
| PonyExl3 | Expert decode and segmented prompt kernel structure; unmodified CPU reference files | [Apache 2.0](third_party/licenses/PonyExl3-Apache-2.0.txt), [repository](https://github.com/beamivalice/PonyExl3), [source revision](https://github.com/beamivalice/PonyExl3/tree/8e7fa6b1556f59fc669e25087903b279b9b0346f), [original NOTICE](third_party/ponyexl3/NOTICE) |
| exllamav3 | EXL3 bit packing, MCG codebook, permutations, and Hadamard transforms | [MIT](third_party/licenses/exllamav3-MIT.txt), [repository](https://github.com/turboderp-org/exllamav3), [source revision](https://github.com/turboderp-org/exllamav3/tree/c5d9c657966ffeeaa9353f0cc899f18629da4a13) |
| vLLM | DeepSeek V3.2 target/MTP normalization and input equations | [Apache 2.0](third_party/licenses/vLLM-Apache-2.0.txt), [repository](https://github.com/vllm-project/vllm), [MTP model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_mtp.py) |
| Z.AI / GLM authors | Base GLM-5.3 model | [Hugging Face model](https://huggingface.co/zai-org/GLM-5.3-BF16), [model license](https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE) |
| KindlingAI / EXLR8 model publisher | Prepared EXLR8 quantization, E4M3-rounded values, layout, and up1 manifest | [Hugging Face checkpoint and model card](https://huggingface.co/kindlingai/glm-5.3-exlr8-k2-k3-k4); quantization additions are credited separately from this runtime |

## Local changes

- `exlr8/model.py`: load EXLR8 packed operators, preserve GLM attention and routing, and select the tested decode paths.
- `exlr8/trellis.py` and expert kernels: apply the EXLR8 E4M3 rounding and stored expert layout to the upstream EXL3 codec and PonyExl3 kernel structure.
- Dense K4 and attention lookup kernels: retain MLX arithmetic order while changing weight loads to the exact local representation.
- Residual-add/RMS kernel: retain the native MLX reduction and round the residual sum to BF16 before normalization.
- Loader, cache, CLI, integration tests, runtime profile, and kernel changes: MLXLR8 / EXLR8 work.

Modified upstream kernels carry source and change notices in their files. The small PonyExl3 reference snapshot in `third_party/ponyexl3/ref` is unmodified. Its original NOTICE and license are retained.

## Distribution

Original MLXLR8 code is Apache 2.0. MIT and Apache notices for upstream work remain in [NOTICE](NOTICE), the source headers, and `third_party`. The Python wheel includes the Metal sources and license texts.

Model weights, private prompts, captured tensor fixtures, and machine-specific benchmark logs are outside this code release. The model's GLM-5.3 license and CC BY 4.0 quantization terms remain separate from the runtime's Apache license.
