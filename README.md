# LLM Verimlilik Teknikleri: Katman Katman Transformer Optimizasyon Rehberi

Büyük dil modellerinin (LLM) verimli eğitimi ve çıkarımı için geliştirilen teknikler, **bellek kullanımını 10 kata kadar azaltırken throughput'u 2-6 kat artırabilmektedir**. FlashAttention, PagedAttention ve GQA gibi yenilikler modern LLM sistemlerinin temelini oluştururken, quantization ve speculative decoding gibi yöntemler üretim dağıtımlarını dönüştürmüştür. Bu rapor, decoder-only Transformer mimarileri için katman katman optimizasyon tekniklerini kapsamlı şekilde ele almaktadır.

---

## 1. Transformer Efficiency Blueprint

Aşağıda her katman için temel verimlilik tekniklerinin özet haritası sunulmaktadır:

| Katman | Temel Optimizasyonlar | Karmaşıklık Değişimi | Ana Hedef |
|--------|----------------------|---------------------|-----------|
| **A) Tokenization + Embedding** | BPE/SentencePiece, Tied Embeddings, Low-rank factorization, MegaByte | - | Bellek, parametre azaltma |
| **B) Normalization** | RMSNorm, Fused kernels, Pre-LN vs Post-LN, DeepNorm | O(7d)→O(3d) | Latency, training stability |
| **C) Attention QKV** | LoRA, MQA, GQA, Pruning, Quantized projections | KV: h×→1× (MQA) | Bellek, fine-tuning efficiency |
| **D) Attention Kernel** | FlashAttention 1/2/3, FlashDecoding, Online softmax | O(n²) mem→O(n) | Latency, memory bandwidth |
| **E) Long Context** | Sparse attention, Ring Attention, RoPE, ALiBi, StreamingLLM | O(n²)→O(n) veya O(n√n) | Uzun dizi desteği |
| **F) KV Cache** | PagedAttention, KIVI/KVQuant, H2O eviction, Prefix caching | %95 sıkıştırma | Bellek kullanımı, throughput |
| **G) FFN/MLP** | SwiGLU, Parallel Attn+FFN, MoE, Fused kernels | Conditional compute | Model kapasitesi, hız |
| **H) Residual/Activation** | Activation checkpointing, LayerDrop, RMSNorm | O(n)→O(√n) bellek | Training memory |
| **I) Model-wide** | Quantization, Pruning, Speculative decoding, Distillation | 3-4× sıkıştırma | Tüm metrikler |
| **J) Systems/Serving** | vLLM, TensorRT-LLM, torch.compile, continuous batching | 36× throughput | Production deployment |

---

## 2. Bölüm A: Tokenization + Embedding + LM Head

Tokenization ve embedding katmanları, model parametrelerinin **%10-20'sini** oluşturabilir ve verimlilik açısından kritik öneme sahiptir.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **BPE (Byte Pair Encoding)** | 2016 | ACL | Vocab boyutu, OOV | En sık karakter çiftlerini iteratif birleştirme ile subword vocabulary oluşturma | Büyük vocab = daha fazla parametre | GPT-2/3/4, RoBERTa | [ACL'16](https://aclanthology.org/P16-1162/) |
| **SentencePiece** | 2018 | EMNLP | Çok dilli, inference hızı | Dil-bağımsız tokenizer; BPE ve Unigram destekli; lossless encode/decode | Domain-bazlı vocab tuning gerekli | LLaMA, T5, mT5 | [GitHub](https://github.com/google/sentencepiece) |
| **Tied Embeddings** | 2016 | EACL'17 | Bellek, parametre | Input embedding ve output softmax ağırlıklarını paylaşma; ~%50 embedding parametre azaltımı | Embedding dim = hidden dim zorunluluğu | GPT-2, BERT, T5 | [arXiv](https://arxiv.org/abs/1608.05859) |
| **MegaByte** | 2023 | arXiv | Throughput, uzun dizi | Tokenization'ı ortadan kaldırır; byte dizilerini sabit patch'lere böler; global+local transformer | Sabit patch suboptimal; 4× daha uzun diziler | Araştırma | [arXiv](https://arxiv.org/abs/2305.07185) |
| **Byte Latent Transformer** | 2024 | arXiv | Compute scaling | Entropy-tabanlı dinamik patching; byte seviyesinde BPE scaling trendleriyle eşleşen ilk mimari | Küçük entropy modeli gerekli | Araştırma | [arXiv](https://arxiv.org/abs/2412.09871) |
| **SpaceByte** | 2024 | NeurIPS | Training FLOPs | Whitespace-tabanlı dinamik patching; kelime sınırlarında büyük transformer blokları | Dile bağımlı patching | Araştırma | [arXiv](https://arxiv.org/abs/2404.14408) |
| **TensorGPT** | 2024 | arXiv | Edge deployment | Tensor-Train Decomposition ile training-free embedding sıkıştırma; 2× compression | Yüksek oranlarında perplexity artışı | Edge cihazlar | [arXiv](https://arxiv.org/abs/2307.00526) |
| **Low-Rank Embedding** | 2019 | AAAI | %90 compression | Eğitim sırasında low-rank matrix factorization; quantization'dan üstün | Retraining gerekli | MobileLLM | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4578) |
| **LightToken** | 2024 | KDD | 25× compression | SVD + residual binary autoencoder kombinasyonu; task/model agnostic | %6 accuracy kaybı @103× | BERT, RoBERTa | [Amazon Science](https://www.amazon.science/blog/compressing-token-embedding-matrices-for-language-models) |

**Pratik Notlar:** LLaMA serisi tied embeddings kullanmaz (daha büyük vocab için). Vocabulary boyutu genellikle **32K-128K** arasında optimize edilir. Byte-level modeller aktif araştırma alanı olup henüz production'da yaygın değildir.

---

## 3. Bölüm B: Normalization

Normalization katmanları training stability ve inference latency'yi doğrudan etkiler. Modern LLM'ler **RMSNorm + Pre-LN** kombinasyonunu tercih etmektedir.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **RMSNorm** | 2019 | NeurIPS | %7-64 daha hızlı | Mean-centering'i kaldırır; sadece RMS ile normalize eder; O(7d)→O(3d) | Re-centering invariance kaybı | LLaMA, Mistral, Gemma | [arXiv](https://arxiv.org/abs/1910.07467) |
| **Pre-LayerNorm** | 2020 | ICML | Training stability | Sublayer'dan önce normalization; warmup gerektirmez; 2× daha hızlı convergence | Post-LN'e göre küçük final performance kaybı | GPT-2/3, LLaMA | [arXiv](https://arxiv.org/abs/2002.04745) |
| **DeepNorm** | 2022 | arXiv | Çok derin modeller | Residual bağlantılarını modifiye eden normalization; 1000+ katmana ölçeklenir | Strict constraint undertrain edebilir | 200-layer 3.2B | [arXiv](https://arxiv.org/abs/2203.00555) |
| **Sub-LN (Magneto)** | 2023 | ICML | Training stability | Her sublayer içinde ekstra LayerNorm; DeepNet-derived initialization | Ek normalization overhead | TorchScale | [arXiv](https://arxiv.org/abs/2210.06423) |
| **QKNorm** | 2020 | EMNLP | Attention stability | Q ve K matrislerine ℓ² normalization; softmax saturation önler | ~%10 throughput azalma | Gemma, OLMoE | [ACL](https://aclanthology.org/2020.findings-emnlp.379/) |
| **Fused LayerNorm/RMSNorm** | 2019-24 | Industry | 2-10× speedup | Tüm normalization op'larını tek CUDA kernel'e birleştirir | Hardware-specific | NVIDIA Apex, Megatron | [Apex](https://nvidia.github.io/apex/layernorm.html) |
| **FlashNorm** | 2024 | arXiv | RMS bottleneck yok | RMS scaling'i sonraki linear layer'a fuse eder; pipeline stall elimine eder | Weight fusion gerekli | Llama, Mistral | [arXiv](https://arxiv.org/abs/2407.09577) |
| **DyT (Dynamic Tanh)** | 2025 | arXiv | Latency | Normalization'ı tanh(α·x)·γ + β ile değiştirir; reduction op yok | Architecture-specific α init | Araştırma | [arXiv](https://arxiv.org/abs/2503.10622) |
| **HybridNorm** | 2025 | arXiv | Stability + perf | QKV normalization + Post-Norm FFN kombinasyonu | Daha karmaşık placement | Araştırma | [arXiv](https://arxiv.org/abs/2503.04598) |

**Pratik Notlar:** RMSNorm artık standart; LLaMA, Mistral, Falcon hepsinde kullanılıyor. Fused kernel implementasyonları üretimde **zorunlu** - PyTorch 2.0+ ve Triton ile native destek mevcut.

---

## 4. Bölüm C: Attention QKV Projections

QKV projection katmanları fine-tuning ve KV cache verimliliği için kritik optimizasyon noktalarıdır.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **LoRA** | 2021 | ICLR'22 | Training memory 3×↓ | Frozen weights + trainable low-rank matrices (W=W₀+BA); 10,000× daha az trainable param | Rank selection heuristic; sadece adaptation | HuggingFace PEFT, tüm LLM fine-tuning | [arXiv](https://arxiv.org/abs/2106.09685) |
| **LoRA+** | 2024 | arXiv | Training speed 2×↑ | A ve B matrislerine farklı learning rate; büyük modellerde optimal | Ekstra hyperparameter | LoRA geliştirmesi | [arXiv](https://arxiv.org/abs/2402.12354) |
| **Multi-Query Attention (MQA)** | 2019 | arXiv | KV cache 12×↓ | Tek K/V head tüm Q head'ler için paylaşılır | Quality degradation olası | PaLM, T5 variants | Shazeer 2019 |
| **Grouped-Query Attention (GQA)** | 2023 | EMNLP | MHA-MQA arası | G grup K/V head; MHA→GQA uptrain %5 compute ile | MHA'ya göre küçük quality kaybı | **LLaMA-2/3, Mistral, Qwen2** | [arXiv](https://arxiv.org/abs/2305.13245) |
| **Linformer** | 2020 | arXiv | O(n²)→O(n) | K/V'yi düşük boyuta project eder; attention matrix low-rank | Approximation; akademik | Araştırma baseline | [arXiv](https://arxiv.org/abs/2006.04768) |
| **Attention Head Pruning** | 2019 | ACL | Parametre azaltma | %40 head prunable; differentiable L0 regularization | Retraining gerekli | Araştırma | [arXiv](https://arxiv.org/abs/1905.09418) |
| **SmoothQuant QKV** | 2022 | ICML'23 | W8A8 INT8 | Quantization difficulty'yi activation'dan weight'e transfer eder | INT8 only | FasterTransformer, Intel NC | [arXiv](https://arxiv.org/abs/2211.10438) |
| **GPTQ QKV** | 2022 | ICLR'23 | Memory 3-4×↓ | Second-order bilgi ile post-training INT4/INT8 quantization | Calibration data gerekli | **vLLM, llama.cpp, HuggingFace** | [arXiv](https://arxiv.org/abs/2210.17323) |

**Pratik Notlar:** GQA modern LLM'lerin standardı haline geldi - LLaMA-2 ve sonrası tüm büyük modeller kullanıyor. LoRA fine-tuning için de-facto standart; QLoRA ile 4-bit base model üzerinde çalışabilir.

---

## 5. Bölüm D: Attention Kernel Optimizasyonları

FlashAttention ailesi, attention hesaplamasında **devrim niteliğinde** iyileştirmeler sağlamıştır.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Performans | Trade-off | Kaynak |
|--------|-----|-------|-------|-----------|------------|-----------|--------|
| **FlashAttention** | 2022 | NeurIPS | Memory O(n²)→O(n) | IO-aware tiling; HBM-SRAM arası memory transfer minimize; online softmax | BERT 15%↑, GPT-2 3×↑ | GPU-specific CUDA kernels | [arXiv](https://arxiv.org/abs/2205.14135) |
| **FlashAttention-2** | 2023 | ICLR'24 | 2× FA1 üzeri | Gelişmiş work partitioning; sequence boyunca paralellik; warp-level optimization | A100'de 225 TFLOPs (%72 MFU) | Ampere/Ada/Hopper specific | [arXiv](https://arxiv.org/abs/2307.08691) |
| **FlashAttention-3** | 2024 | NeurIPS | 1.5-2× FA2 üzeri | Warp-specialization + TMA asynchrony; FP8 block quantization + Hadamard transform | H100'de **740 TFLOPs FP16, 1.2 PFLOPs FP8** | Hopper-only | [arXiv](https://arxiv.org/abs/2407.08608) |
| **Flash-Decoding** | 2023 | Stanford Blog | 8× uzun dizi | K/V sequence üzerinde yeni parallelization; batch=1 için %\<1 utilization sorunu çözer | Uzun dizilerde 8× speedup | Ekstra reduction step | [Stanford](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) |
| **FlashDecoding++** | 2024 | MLSys | 4.86× HF üzeri | Async softmax, flat GEMM optimization, heuristic dataflow | 1.37× Flash-Decoding üzeri | Architecture-specific tuning | [arXiv](https://arxiv.org/abs/2311.01282) |
| **Online Softmax** | 2018 | arXiv | Memory BW 1.3×↓ | Tek pass'te softmax normalizer hesaplama; telescoping sum correction | FlashAttention'ın temeli | Careful numerical impl | [arXiv](https://arxiv.org/abs/1805.02867) |
| **xFormers** | 2022 | Meta | Memory, latency | Modüler attention operators; CUTLASS-based; P100+ GPU desteği | Stable Diffusion %40↑ | FA'dan daha az optimize | [GitHub](https://github.com/facebookresearch/xformers) |
| **Triton Fused Attention** | 2022+ | OpenAI | Portability | Python DSL ile FA-2 implementasyonu; AMD ROCm desteği | A100'de ~150-160 TFLOPs | JIT overhead; peak perf düşük | [Triton](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) |
| **cuDNN SDPA** | 2024 | NVIDIA | Production | Closed-source optimal SDPA; FP8/BF16/FP16; otomatik algorithm seçimi | H200'de **1.2 PFLOPs FP8** | NVIDIA-only | [NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/) |
| **PyTorch SDPA** | 2023 | PyTorch | Ease of use | Native API; FA/xFormers/C++ backend'e otomatik dispatch | Default PyTorch 2.0+ | Limited mask support | [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) |

**Pratik Notlar:** FlashAttention-2 şu an endüstri standardı. H100 için FlashAttention-3 veya cuDNN tercih edilmeli. PyTorch SDPA kullanımı önerilir - backend'i otomatik seçer.

---

## 6. Bölüm E: Long Context Attention Patterns

Uzun dizi desteği için sparse, linear ve yapısal attention yaklaşımları geliştirilmiştir.

| Teknik | Yıl | Venue | Karmaşıklık | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------------|-----------|-----------|----------|--------|
| **Sparse Transformer** | 2019 | arXiv | O(n²)→O(n√n) | Strided + fixed attention patterns ile sparse factorization | Approximate; bazı global bağımlılıklar kaçırılabilir | Temel çalışma | [arXiv](https://arxiv.org/abs/1904.10509) |
| **Longformer** | 2020 | arXiv | O(n²)→O(n) | Sliding window + task-motivated global tokens; 4096 token | Task-specific global token placement | HuggingFace | [arXiv](https://arxiv.org/abs/2004.05150) |
| **BigBird** | 2020 | NeurIPS | O(n²)→O(n) | Random + window + global attention; Turing complete kanıtlanmış | Ratio tuning gerekli | Genomics, QA | [arXiv](https://arxiv.org/abs/2007.14062) |
| **Performer (FAVOR+)** | 2021 | ICLR | O(n²)→O(n) | Random feature approximation ile softmax; sparsity varsayımı yok | Kısa dizilerde suboptimal | Google | [arXiv](https://arxiv.org/abs/2009.14794) |
| **RoPE** | 2021/24 | Neurocomputing | Length flexibility | Rotation matrix ile absolute+relative position encoding | Extreme extrapolation'da struggle | **LLaMA, Qwen, Gemma, GPT-J** | [arXiv](https://arxiv.org/abs/2104.09864) |
| **ALiBi** | 2022 | ICLR | Length extrapolation | Static linear bias; öğrenilecek parametre yok; recency bias | Tüm pozisyonlara eşit ağırlık gereken tasklarda zayıf | MPT, BLOOM | [arXiv](https://arxiv.org/abs/2108.12409) |
| **StreamingLLM** | 2023 | ICLR'24 | Infinite inference | "Attention sink" fenomeni; sink tokens + sliding window | Gerçek context uzatmaz; sadece fluency | TensorRT-LLM, HuggingFace | [arXiv](https://arxiv.org/abs/2309.17453) |
| **Ring Attention** | 2023 | NeurIPS | Distributed | Ring topology'de attention dağıtımı; communication/compute overlap | Multi-device gerekli | Large World Model | [arXiv](https://arxiv.org/abs/2310.01889) |
| **YaRN** | 2023 | ICLR'24 | 10× az token | NTK-by-parts + attention temperature scaling; 128K+ context | Fine-tuning gerekli | NousResearch, vLLM | [arXiv](https://arxiv.org/abs/2309.00071) |

**Pratik Notlar:** RoPE + FlashAttention kombinasyonu modern LLM'ler için standart. Context extension için YaRN tercih ediliyor. StreamingLLM gerçek context uzatmaz, sadece fluent generation sağlar.

---

## 7. Bölüm F: KV Cache Optimizasyonları

KV cache, autoregressive decoding'de **bellek darboğazının** ana kaynağıdır.

| Teknik | Yıl | Venue | Bellek Etkisi | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|---------------|-----------|-----------|----------|--------|
| **MHA (Baseline)** | 2017 | NeurIPS | O(n×L×h×d) | Her head için bağımsız K/V | Maximum expressivity, max memory | Original Transformer | [arXiv](https://arxiv.org/abs/1706.03762) |
| **Multi-Query Attention** | 2019 | arXiv | KV h×↓ | Tek K/V head tüm Q head'ler için | Quality degradation riski | PaLM, T5 | [arXiv](https://arxiv.org/abs/1911.02150) |
| **Grouped-Query Attention** | 2023 | EMNLP | KV (h/g)×↓ | G grup K/V; %5 compute ile MHA→GQA uptrain | MHA'dan az quality kaybı | **LLaMA-2/3, Mistral, Qwen2** | [arXiv](https://arxiv.org/abs/2305.13245) |
| **PagedAttention / vLLM** | 2023 | SOSP | <%4 waste | OS virtual memory paging; non-contiguous blocks; copy-on-write | Block size trade-off | **Endüstri standardı** | [arXiv](https://arxiv.org/abs/2309.06180) |
| **KIVI** | 2024 | ICML | 2.6× mem↓ | Tuning-free asymmetric 2-bit; key per-channel, value per-token quantization | Residual tokens FP16 | HuggingFace KV quant | [arXiv](https://arxiv.org/abs/2402.02750) |
| **KVQuant** | 2024 | NeurIPS | 10M context @ 8 GPU | Per-channel key, pre-RoPE, non-uniform quantization, dense-and-sparse | Daha karmaşık quant | 1.7× kernel speedup | [arXiv](https://arxiv.org/abs/2401.18079) |
| **H2O (Heavy-Hitter Oracle)** | 2023 | NeurIPS | %20 KV yeterli | %95 sparsity gözlemi; cumulative attention score ile eviction | Önemli token kaybı riski | 29× throughput↑ | [arXiv](https://arxiv.org/abs/2306.14048) |
| **Scissorhands** | 2023 | NeurIPS'24 | Configurable | "Persistence of importance" hipotezi; batch eviction | Importance stability varsayımı | Baseline | [arXiv](https://arxiv.org/abs/2305.17118) |
| **Prefix Caching** | 2023-24 | Various | Repeated prompt latency↓ | Ortak prefix'ler için KV state cache; system prompt reuse | Cache invalidation complexity | vLLM, SGLang (RadixAttention) | vLLM docs |

**Pratik Notlar:** GQA + PagedAttention kombinasyonu modern LLM serving'in temeli. KV cache quantization aktif araştırma alanı - KIVI production-ready. H2O gibi eviction politikaları uzun context için kritik.

---

## 8. Bölüm G: FFN/MLP Optimizasyonları

FFN katmanları toplam parametrelerin **~%66'sını** (GQA ile %80) oluşturur.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **SwiGLU** | 2020 | arXiv | Quality at same params | Gated Linear Unit + Swish: SwiGLU(x,W,V) = Swish(xW) ⊗ xV | 3 matris (2 yerine); 8/3× dim | **LLaMA, PaLM, Mistral, Falcon** | [arXiv](https://arxiv.org/abs/2002.05202) |
| **GEGLU/ReGLU** | 2020 | arXiv | Quality variants | GLU with GELU/ReLU gating | SwiGLU ile benzer trade-off | Araştırma alternatifleri | [arXiv](https://arxiv.org/abs/2002.05202) |
| **FFN Intermediate 8/3×** | 2023 | LLaMA | Parameter efficiency | SwiGLU ile 4× yerine ~2.68× intermediate; multiple_of=256 için hardware-efficient | Kapasite vs. parametre trade-off | LLaMA (11008 @ 4096 hidden) | [HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/llama) |
| **Parallel Attn+FFN** | 2021 | EleutherAI | %15 training throughput↑ | output = x + Attn(x) + FFN(x); tek all-reduce | Küçük accuracy impact | GPT-J, GPT-NeoX, PaLM | [Cerebras Blog](https://www.cerebras.ai/blog/how-to-harness-the-predictive-power-of-gpt-j) |
| **Fused FFN Kernels** | 2020-24 | NVIDIA | 2× throughput | MatMul + bias + activation tek kernel; SwiGLU fusion | CUDA-specific | FasterTransformer, TensorRT-LLM | [GitHub](https://github.com/NVIDIA/FasterTransformer) |
| **Mixture-of-Experts (MoE)** | 2021 | JMLR | Sublinear compute | Birden fazla FFN "expert"; top-k routing; sadece k expert aktif | Load balancing; memory for all experts | Mixtral, Switch, DeepSeek | [arXiv](https://arxiv.org/abs/2101.03961) |
| **FFN Fusion** | 2025 | arXiv | 1.71× cost↓ | Sequential FFN'leri parallel'e dönüştürme; attention pruning ile | Büyük ölçeklerde (49B+) etkili | Llama-Nemotron | [arXiv](https://arxiv.org/html/2503.18908v1) |
| **DaSS (N:M Sparsity)** | 2024 | arXiv | Hardware-friendly pruning | SwiGLU için activation-aware N:M sparsity metric | Structured sparsity kısıtları | LLaMA2, Mistral | [arXiv](https://arxiv.org/html/2405.01943v1) |

**Pratik Notlar:** SwiGLU artık standart - tüm modern LLM'ler kullanıyor. MoE büyük modeller için cost-effective (Mixtral 8x7B = 70B performans, 13B active params). Fused kernel'lar production'da zorunlu.

---

## 9. Bölüm H: Residual, Dropout ve Activation Functions

Bu katmanlar training memory ve stability için kritik optimizasyon noktalarıdır.

| Teknik | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **Activation Checkpointing** | 2016 | arXiv | Memory O(n)→O(√n) | Her √n layer'da checkpoint; backprop'ta recompute | %20-30 training slowdown | **Universal** (GPT-3, LLaMA, BERT) | [arXiv](https://arxiv.org/abs/1604.06174) |
| **Pre-LN Architecture** | 2020 | ICML | Training stability | x + F(LN(x)); warmup gerektirmez; 2× convergence | Post-LN'e göre küçük final quality kaybı | **GPT-2/3, LLaMA** | [arXiv](https://arxiv.org/abs/2002.04745) |
| **RMSNorm** | 2019 | NeurIPS | ~%5 faster | Mean-centering yok; O(7d)→O(3d) | LayerNorm'dan farklı davranış | **LLaMA, Mistral, Falcon** | [arXiv](https://arxiv.org/abs/1910.07467) |
| **Reversible Residuals** | 2020 | ICLR | O(1) activation memory | RevNet-style split streams; backprop'ta reconstruction | Architectural değişiklik; serial dependency | Reformer | [arXiv](https://arxiv.org/abs/2001.04451) |
| **LayerDrop** | 2019 | ICLR'20 | Flexible inference depth | Training'de random layer drop; inference'da herhangi depth | Sub-network quality kaybı | RoBERTa-LayerDrop | [arXiv](https://arxiv.org/abs/1909.11556) |
| **Stochastic Depth** | 2016 | ECCV | Training speedup | Random layer skip with identity; 1200+ layer training | Drop rate scheduling gerekli | CaiT, DeiT | [arXiv](https://arxiv.org/abs/1603.09382) |
| **GELU (tanh approx)** | 2016 | arXiv | Model quality | GELU(x) = x·Φ(x); tanh approximation 6-10× faster | ReLU'dan yavaş (optimize edilmeden) | BERT, GPT-2/3 | [arXiv](https://arxiv.org/abs/1606.08415) |
| **Swish/SiLU** | 2017 | arXiv | NAS-discovered | Swish(x) = x·σ(x); smooth, self-gated | ReLU'dan biraz daha pahalı | SwiGLU component, LLaMA | [arXiv](https://arxiv.org/abs/1710.05941) |
| **No Dropout (Pre-training)** | 2020 | arXiv | Quality improvement | Large-scale pre-training'de dropout=0 daha iyi | Küçük dataset'lerde overfit riski | **LLaMA, GPT-3** | [arXiv](https://arxiv.org/abs/2002.05202) |
| **Selective Checkpointing** | 2024 | PyTorch | Better mem-speed trade-off | Expensive ops (matmul) sakla, cheap ops (activation) recompute | Konfigürasyon karmaşıklığı | PyTorch 2.0+ | [PyTorch Blog](https://pytorch.org/blog/activation-checkpointing-techniques/) |

**Pratik Notlar:** Activation checkpointing evrensel olarak kullanılır. Pre-LN + RMSNorm modern standart. Pre-training'de dropout=0 tercih edilir; fine-tuning'de 0.1 kullanılabilir.

---

## 10. Bölüm I: Model-Wide Optimizasyonlar

Bu teknikler tüm katmanlarda compute/memory azaltımı sağlar.

### 10.1 Mixture of Experts (MoE)

| Teknik | Yıl | Venue | Impact | Ana Fikir | Kullanım | Kaynak |
|--------|-----|-------|--------|-----------|----------|--------|
| **Switch Transformer** | 2021 | JMLR | 7× pre-training speedup | Top-1 expert routing; 1.6T parametreye scale | Foundation | [arXiv](https://arxiv.org/abs/2101.03961) |
| **GShard** | 2020 | arXiv | 600B params | Top-2 gating; automatic sharding | Google Translate | [arXiv](https://arxiv.org/abs/2006.16668) |
| **Mixtral 8x7B** | 2024 | arXiv | LLaMA2 70B = 6× faster | 8 expert, top-2; 46.7B total, 12.9B active | **Apache 2.0, production** | [arXiv](https://arxiv.org/abs/2401.04088) |
| **DeepSeekMoE** | 2024 | ACL | %40 compute = eşit perf | Fine-grained experts + shared expert isolation | DeepSeek-V2/V3 | [arXiv](https://arxiv.org/abs/2401.06066) |

### 10.2 Quantization

| Teknik | Yıl | Venue | Impact | Ana Fikir | Kullanım | Kaynak |
|--------|-----|-------|--------|-----------|----------|--------|
| **GPTQ** | 2022 | ICLR'23 | 3-4× memory↓, 3.25× speedup | Second-order Hessian ile layer-wise INT4/8 quantization | **vLLM, llama.cpp, HuggingFace** | [arXiv](https://arxiv.org/abs/2210.17323) |
| **AWQ** | 2023 | MLSys'24 Best | 3×+ speedup, mobile deployment | %1 salient weights (activation-based); per-channel scaling | TinyChat, edge | [arXiv](https://arxiv.org/abs/2306.00978) |
| **SmoothQuant** | 2022 | ICML'23 | W8A8, 1.56× speedup | Activation→weight difficulty transfer | INT8 GEMM acceleration | [arXiv](https://arxiv.org/abs/2211.10438) |
| **QLoRA** | 2023 | NeurIPS | 65B on 48GB GPU | 4-bit NF4 + LoRA + double quantization | **Democratized LLM fine-tuning** | [arXiv](https://arxiv.org/abs/2305.14314) |

### 10.3 Pruning

| Teknik | Yıl | Venue | Impact | Ana Fikir | Kullanım | Kaynak |
|--------|-----|-------|--------|-----------|----------|--------|
| **SparseGPT** | 2023 | ICML | %60 sparsity, min perplexity↑ | Second-order pruning; <4.5h on A100 for 175B | 2:4 structured patterns | [arXiv](https://arxiv.org/abs/2301.00774) |
| **Wanda** | 2023 | ICLR'24 | Single forward pass | \|weight\| × \|\|activation\|\| metric; ~10 lines code | Hızlı iterasyon | [arXiv](https://arxiv.org/abs/2306.11695) |

### 10.4 Speculative Decoding

| Teknik | Yıl | Venue | Impact | Ana Fikir | Kullanım | Kaynak |
|--------|-----|-------|--------|-----------|----------|--------|
| **Speculative Decoding** | 2023 | ICML | 2-3× speedup (lossless) | Draft model γ token üretir; target parallel verify | vLLM, TensorRT-LLM | [arXiv](https://arxiv.org/abs/2211.17192) |
| **Medusa** | 2024 | ICML | 2.2-3.6× speedup | Multiple prediction heads; tree-based verification | FastChat, vLLM | [arXiv](https://arxiv.org/abs/2401.10774) |
| **EAGLE** | 2024 | ICML | 3-6.5× speedup | Feature-level drafting; #1 on Spec-Bench | vLLM, TensorRT-LLM, SGLang | [arXiv](https://arxiv.org/abs/2401.15077) |

### 10.5 Early Exit & Distillation

| Teknik | Yıl | Venue | Impact | Ana Fikir | Kullanım | Kaynak |
|--------|-----|-------|--------|-----------|----------|--------|
| **CALM** | 2022 | NeurIPS | 3× speedup | Confidence-based early exit; LTT framework | T5 | [arXiv](https://arxiv.org/abs/2207.07061) |
| **DistilBERT** | 2019 | NeurIPS WS | %40 smaller, %60 faster | Triple loss distillation; half layers | Foundation work | [arXiv](https://arxiv.org/abs/1910.01108) |

---

## 11. Bölüm J: Systems ve Serving Stack

Production LLM deployment için sistem seviyesi optimizasyonlar.

| Sistem | Yıl | Venue | Hedef | Ana Fikir | Trade-off | Kullanım | Kaynak |
|--------|-----|-------|-------|-----------|-----------|----------|--------|
| **vLLM** | 2023 | SOSP | Memory, throughput | PagedAttention; continuous batching; <%4 waste | Block size tuning | **Endüstri standardı** | [arXiv](https://arxiv.org/abs/2309.06180) |
| **TensorRT-LLM** | 2023 | NVIDIA | Latency, throughput | Custom kernels, FP8/FP4, in-flight batching | NVIDIA-only | H100/H200 production | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **SGLang** | 2024 | NeurIPS | Structured generation | RadixAttention (52-74% cache hit), FSM optimization | Learning curve | Chatbot Arena | [arXiv](https://arxiv.org/abs/2312.07104) |
| **llama.cpp** | 2023 | GitHub | Portability, CPU | C/C++ pure impl; GGUF format; 1.5-8 bit quant | Peak GPU perf düşük | Ollama, LM Studio | [GitHub](https://github.com/ggml-org/llama.cpp) |
| **TGI** | 2022 | HuggingFace | Production serving | Rust/Python; Flash Attention, continuous batching | Maintenance mode | HuggingChat | [GitHub](https://github.com/huggingface/text-generation-inference) |
| **Orca** | 2022 | OSDI | 36.9× throughput | Iteration-level scheduling, selective batching | KV cache pressure↑ | Foundation for all | [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu) |
| **torch.compile** | 2023 | PyTorch | %43 avg speedup | TorchDynamo + TorchInductor; Triton codegen | First-run compilation | PyTorch 2.0+ | [PyTorch](https://pytorch.org/get-started/pytorch-2-x/) |
| **Triton** | 2019 | OpenAI | Productivity | Python GPU DSL; block-based programming | Peak perf vs CUDA | TorchInductor, vLLM | [Triton](https://triton-lang.org) |
| **DeepSpeed ZeRO** | 2020 | SC | Training memory | Optimizer/gradient/param sharding; CPU/NVMe offload | Communication overhead | GPT-3 training | [arXiv](https://arxiv.org/abs/1910.02054) |
| **PyTorch FSDP** | 2023 | arXiv | Memory-efficient DDP | ZeRO-3 style sharding; native PyTorch | Wrapping policy complexity | GPT-175B: 159 TFLOPS | [arXiv](https://arxiv.org/abs/2304.11277) |
| **FlashInfer** | 2025 | MLSys Best | Kernel flexibility | JIT attention kernels; block-sparse format | JIT latency | vLLM, SGLang | [arXiv](https://arxiv.org/abs/2501.01005) |
| **Marlin** | 2024 | vLLM | 4-bit inference | Optimized INT4/INT8/FP8 CUDA kernels | Ampere+ (SM 8.0+) | vLLM default quant | [GitHub](https://github.com/vllm-project/vllm) |

---

## 12. Top 15 Must-Know Papers ve Projeler

Aşağıdaki çalışmalar LLM verimliliği için **temel taşlar** niteliğindedir:

| # | Çalışma | Yıl | Etki | Neden Önemli |
|---|---------|-----|------|--------------|
| 1 | **FlashAttention** | 2022 | Attention memory O(n²)→O(n) | Modern LLM training/inference'ın temeli; tüm major framework'lerde |
| 2 | **PagedAttention / vLLM** | 2023 | KV cache waste %60-80→<%4 | LLM serving standardı; 2-4× throughput artışı |
| 3 | **GQA** | 2023 | KV cache h×↓ | LLaMA-2+ tüm modern LLM'lerin standardı |
| 4 | **LoRA** | 2021 | Training memory 3×↓ | LLM fine-tuning demokratizasyonu; PEFT foundation |
| 5 | **GPTQ** | 2022 | Model size 3-4×↓ | Post-training quantization standardı |
| 6 | **AWQ** | 2023 | Edge deployment mümkün | Activation-aware quant; mobile LLM |
| 7 | **QLoRA** | 2023 | 65B on 48GB | 4-bit fine-tuning; accessibility revolution |
| 8 | **Mixtral/MoE** | 2023/24 | 6× inference efficiency | Sparse compute paradigm shift |
| 9 | **Speculative Decoding** | 2023 | 2-3× latency↓ (lossless) | Decoding paradigm değişikliği |
| 10 | **EAGLE** | 2024 | 6.5× latency↓ | State-of-the-art speculative |
| 11 | **SwiGLU** | 2020 | Better quality/param ratio | Modern FFN standardı |
| 12 | **RMSNorm** | 2019 | %7-64 faster normalization | LLaMA/Mistral/Gemma standardı |
| 13 | **RoPE** | 2021 | Length flexibility | Position encoding standardı |
| 14 | **DeepSpeed ZeRO** | 2020 | Trillion param training | Distributed training foundation |
| 15 | **torch.compile** | 2023 | %43 avg speedup | PyTorch optimization standardı |

---

## 13. Engineering Playbook

### 13.1 Training Optimization Stack

**Memory Optimization (Öncelik Sırası):**
1. **Activation Checkpointing** - Gradient checkpointing ile bellek %60-85↓ ([PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/))
2. **Mixed Precision (BF16/FP16)** - Memory 2×↓, throughput 2×↑
3. **ZeRO Stage 2/3** veya **FSDP** - Optimizer state sharding
4. **Gradient Accumulation** - Effective batch size artırımı

**Compute Optimization:**
1. **torch.compile** (mode="max-autotune") - %30-50 speedup
2. **FlashAttention-2/3** - Attention dominant workloads için
3. **Fused Kernels** - Transformer Engine, Apex

**Distributed Training:**
- **\<10B params**: Data Parallel + Gradient Checkpointing
- **10-70B params**: FSDP/ZeRO-3 + Tensor Parallelism
- **>70B params**: 3D Parallelism (TP + PP + DP) + ZeRO

### 13.2 Inference/Serving Optimization Stack

**Latency Optimization (Interactive):**
1. **Speculative Decoding** (EAGLE/Medusa) - 2-6× latency↓
2. **FlashAttention-2/3** - Prefill acceleration
3. **Quantization** (AWQ/GPTQ INT4) - Memory bandwidth↓
4. **KV Cache Quantization** (KIVI) - Longer context

**Throughput Optimization (Batch):**
1. **Continuous Batching** - Orca-style scheduling
2. **PagedAttention** - KV cache efficiency
3. **Prefix Caching** - RadixAttention
4. **Dynamic SplitFuse** - Variable sequence handling

**Recommended Stack by Use Case:**

| Use Case | Framework | Quantization | Notes |
|----------|-----------|--------------|-------|
| Research/Prototyping | HuggingFace + torch.compile | None/BF16 | Flexibility öncelikli |
| Production Serving | vLLM veya SGLang | AWQ INT4 | Throughput öncelikli |
| NVIDIA Optimized | TensorRT-LLM | FP8 (H100) | Maximum performance |
| Edge/Mobile | llama.cpp | GGUF Q4_K_M | Portability öncelikli |
| Fine-tuning | HuggingFace + QLoRA | NF4 | Memory efficiency |

### 13.3 Hardware-Specific Notes

**NVIDIA A100 (80GB):**
- FlashAttention-2 optimal
- BF16 training, INT8 inference
- SmoothQuant W8A8 for throughput
- Tensor Core utilization: %70-80 achievable

**NVIDIA H100/H200:**
- FlashAttention-3 veya cuDNN SDPA
- **FP8 training ve inference** - 2× throughput vs BF16
- Transformer Engine entegrasyonu
- TensorRT-LLM + FP8 for production

**Consumer GPUs (RTX 3090/4090):**
- AWQ/GPTQ INT4 zorunlu (>13B models)
- llama.cpp veya ExLlamaV2
- FlashAttention-2 (Triton backend)
- Batch size 1-4 optimal

**AMD MI300X:**
- Triton kernels (vLLM default)
- FlashInfer ROCm support
- HIP-based implementations
- ONNX Runtime optimization

---

## 14. Gaps ve Open Problems (2026)

LLM verimliliği alanında çözülmemiş önemli sorunlar:

1. **Ultra-Long Context Scaling**: 1M+ token context için efficient attention hala çözülmemiş. Ring Attention communication overhead yüksek; sparse attention quality kaybı var.

2. **Hardware-Software Co-design**: FP4/FP6 gibi yeni formatlar için optimal kernel design. Blackwell-specific optimizations (Flash Attention 4) henüz erken aşamada.

3. **Dynamic Sparsity Acceleration**: Unstructured sparsity'nin hardware acceleration'ı hala zayıf. 2:4 structured sparsity kısıtlayıcı.

4. **MoE Load Balancing**: Expert routing instability ve dead experts problemi tam çözülmedi. Communication overhead distributed MoE için darboğaz.

5. **KV Cache Compression Quality**: Aggressive quantization (\<2-bit) ve eviction politikalarının long-range dependency'ye etkisi belirsiz.

6. **Speculative Decoding for Small Draft Models**: Draft model quality/size trade-off optimal değil. Self-speculative approaches (EAGLE) henüz olgunlaşıyor.

7. **Cross-Platform Optimization**: AMD, Intel GPU ve custom accelerator'lar için NVIDIA-comparable performance gap devam ediyor.

8. **Energy Efficiency Metrics**: FLOPs/Watt optimizasyonu için standardize edilmiş benchmark'lar yok. Carbon footprint azaltımı için sistematik yaklaşım eksik.

9. **Activation Memory for Training**: Checkpointing overhead (%20-30 slowdown) hala yüksek. Reversible architectures adoption düşük.

10. **Quantization-Aware Training at Scale**: Post-training quantization dominant; QAT for LLMs henüz pratik değil.

11. **Latency-Throughput Pareto Frontier**: Interactive (low latency) ve batch (high throughput) serving için unified optimal system yok.

12. **Compiler Auto-Optimization**: torch.compile graph breaks; dynamic shapes problematic. Fully automatic optimization henüz achievable değil.

---

## Sonuç

LLM verimliliği **2022-2025 arasında devrim niteliğinde** ilerleme kaydetmiştir. FlashAttention ailesi attention computation'ı dönüştürürken, PagedAttention serving verimliliğini yeniden tanımlamıştır. GQA ve SwiGLU modern mimarilerin standart bileşenleri haline gelirken, quantization (GPTQ, AWQ, QLoRA) model dağıtımını demokratikleştirmiştir.

**En kritik insight**: Verimlilik kazanımları tek bir teknikten değil, katmanlar arası optimizasyon kombinasyonlarından gelir. Örneğin, **H100 + FlashAttention-3 + GQA + PagedAttention + EAGLE + FP8** stack'i, naive implementation'a göre **10-20× throughput** ve **5-10× latency** iyileştirmesi sağlayabilir.

Gelecekte byte-level tokenization, ultra-long context, ve hardware-specific kernel optimization alanlarında önemli gelişmeler beklenmektedir. Production deployment için vLLM/TensorRT-LLM + AWQ/FP8 + speculative decoding kombinasyonu şu an optimal noktadadır.
