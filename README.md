# SuperAI Platform

Enterprise-grade AI Inference Backend with multi-engine routing, OpenAI-compatible API, multimodal pipelines, and Kubernetes-native deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ingress (NGINX + TLS)                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     API Gateway (FastAPI)                        │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │   Auth   │ │Rate Limit │ │ Metrics  │ │  Error Handler   │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    Inference Router                              │
│  Model Registry │ Load Balancer │ Request Queue │ Health Check  │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬───────────────────┘
    │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌──────┐┌─────┐┌──────┐┌──────┐┌────────┐┌────────┐┌──────────┐
│ vLLM ││ TGI ││SGLang││Ollama││TRT-LLM ││LMDeploy││DeepSpeed │
└──────┘└─────┘└──────┘└──────┘└────────┘└────────┘└──────────┘
    │      │      │      │      │      │      │
    └──────┴──────┴──────┴──────┴──────┴──────┘
                       │
              ┌────────▼────────┐
              │   GPU Cluster   │
              │  (NVIDIA A100)  │
              └─────────────────┘
```

## Features

### Multi-Engine Inference Routing
- **vLLM**: PagedAttention, continuous batching, prefix caching (68.7k★)
- **TGI**: HuggingFace ecosystem, Flash Attention 2, production-grade
- **SGLang**: RadixAttention, structured generation, 6.4x throughput boost
- **Ollama**: One-command local deployment, GGUF quantization
- **TensorRT-LLM**: NVIDIA deep optimization, FP8/FP4, kernel fusion
- **LMDeploy**: Dual-engine (TurboMind + PyTorch), KV-cache quantization
- **DeepSpeed**: ZeRO optimization, large-scale distributed inference

### OpenAI-Compatible API
- `/v1/chat/completions` - Chat completion (streaming + non-streaming)
- `/v1/completions` - Text completion
- `/v1/embeddings` - Text embeddings
- `/v1/models` - Model management
- `/v1/audio/transcriptions` - Speech-to-text (Whisper)
- `/v1/audio/speech` - Text-to-speech
- `/v1/images/generations` - Image generation

### Multimodal Pipelines
- **Vision-Language**: Qwen2.5-VL (256K context), LLaVA-NeXT, InternVL3.5
- **Image Generation**: Stable Diffusion XL, FLUX via ComfyUI/diffusers
- **Audio**: Whisper V3 Turbo (sub-second ASR), CosyVoice TTS
- **Video**: Frame sampling + VLM analysis, temporal reasoning

### Specialized Scenarios
- **Code Generation**: DeepSeek-Coder, FIM completion, code review
- **RAG Pipeline**: Chunking, embedding, vector search, cited generation
- **Agent/Function Calling**: ReAct loop, Plan-and-Execute, tool orchestration
- **Batch Inference**: Async parallel processing, priority scheduling

### Enterprise Features
- JWT + API Key authentication
- Sliding window rate limiting (Redis-backed)
- Prometheus metrics + Grafana dashboards
- Structured JSON logging for k8s log aggregation
- Health checks with engine-level granularity
- HPA autoscaling based on request load

## Quick Start

### Docker Compose
```bash
cp .env.example .env
# Edit .env with your HF_TOKEN and settings
cd docker && docker compose up -d
```

### Kubernetes
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### Helm
```bash
helm install superai ./helm \
  --namespace superai \
  --create-namespace \
  --set secrets.hfToken=$HF_TOKEN
```

### Local Development
```bash
pip install -r requirements.txt
SUPERAI_ENVIRONMENT=development SUPERAI_DEBUG=true \
  uvicorn src.app:app --reload --port 8000
```

## API Usage

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-superai-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 2048
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-superai-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Write a poem"}],
    "stream": true
  }'

# RAG query
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer sk-superai-..." \
  -d '{"collection_name": "docs", "question": "What is PagedAttention?"}'
```

## Testing

```bash
pip install pytest pytest-asyncio pytest-cov
pytest tests/ -v --cov=src
```

## License

Apache-2.0