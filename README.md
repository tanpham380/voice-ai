# Voice AI Service — STT → LLM → TTS (streaming)

Dịch vụ giọng nói tiếng Việt chạy độc lập: **thu âm → nhận dạng (STT) → AI tư vấn (LLM) → đọc lại (TTS)**,
với khả năng **streaming thời gian thực** (text và audio phát đồng bộ từng câu).

```
mic / text ──► STT (nghi-stt-v3, sherpa-onnx)
              ──► LLM (OpenAI-compatible: DeepSeek qua AI-Box gateway)
                    ──► TTS (VieNeu v3 Turbo, 48 kHz, GPU)
                          ──► Web Audio (phát stream nối tiếp)
```

Công nghệ: **Python + FastAPI**, chạy bằng **Docker Compose** (hỗ trợ GPU NVIDIA qua WSL2).

---

## ✨ Tính năng

- **Streaming đồng bộ (1 LLM call)**: text và audio được stream cùng lúc qua SSE
  (`event: text` + `event: audio`), không gọi LLM 2 lần → không lặp, không chậm gấp đôi.
- **STT tiếng Việt**: `nghi-stt-v3` (Zipformer transducer, sherpa-onnx), chạy offline.
- **LLM tư vấn**: OpenAI-compatible (mặc định DeepSeek qua AI-Box gateway),
  hỗ trợ streaming token, tắt "thinking" để phản hồi nhanh.
- **TTS VieNeu v3 Turbo**: 48 kHz, 20+ giọng tiếng Việt, chạy trên **GPU CUDA**.
- **Format text trước khi TTS**: tự động loại bỏ markdown, emoji, ký tự đặc biệt,
  chuẩn hóa dấu câu tiếng Việt → giọng đọc tự nhiên, rõ ràng.
- **Điều chỉnh tốc độ nói**: `TTS_SPEED` (0.75–1.5) map sang `max_new_frames` của VieNeu.
- **Web Audio player**: phát PCM16 stream nối tiếp (không chồng chéo → không nháo nhào).
- **Barge-in**: người dùng có thể cắt ngang khi bot đang nói (pipeline mode).

---

## 🚀 Nhanh chóng bắt đầu

### 1. Yêu cầu
- Docker Desktop với **WSL2 backend** + NVIDIA GPU (driver hỗ trợ GPU-PV).
- Model STT/TTS (tải 1 lần):

```bash
python scripts/download_models.py        # STT (nghi-stt-v3) + Silero VAD
python scripts/download_tts_models.py    # VieNeu TTS (hoặc để tự động tải từ HF)
```

### 2. Cấu hình
Copy `.env.example` → `.env` và chỉnh sửa:

```bash
cp .env.example .env
```

Các biến quan trọng:
| Biến | Ý nghĩa | Mặc định |
|------|---------|----------|
| `OPENAI_BASE_URL` | LLM endpoint (OpenAI-compatible) | `https://api.ai-box.vn/v1` |
| `OPENAI_API_KEY` | API key | — |
| `OPENAI_MODEL` | Tên model | `deepseek-v4-flash` |
| `OPENAI_ENABLE_THINKING` | Tắt thinking (0/1) | `0` |
| `TTS_DEFAULT_VOICE` | Giọng mặc định | `Adam` |
| `TTS_SPEED` | Tốc độ nói (1.0 tự nhiên) | `1.05` |
| `TTS_SILENCE_P` | Khoảng nghỉ giữa câu | `0.15` |

### 3. Build & chạy (GPU)
```bash
docker compose build
docker compose up -d
```
Mở **http://localhost:8000** để test voice chat.

> Để chạy CPU-only: sửa `docker-compose.yml` → `GPU: "0"`, `TARGET_STAGE: "cpu"`.

---

## 🔌 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/health` | Trạng thái STT/TTS/AI |
| `POST` | `/v1/chat` | Text → LLM (SSE token) |
| `POST` | `/v1/chat/stream` | **Text → LLM + TTS stream (SSE: text + audio PCM)** |
| `POST` | `/v1/chat/stream-audio` | Text → WAV stream (LLM→TTS) |
| `POST` | `/v1/stt` | Audio → transcript |
| `POST` | `/v1/voice/chat` | Audio → {transcript, answer, audio base64} (batch) |
| `POST` | `/v1/voice/chat/stream` | **Audio → STT → LLM + TTS stream (SSE)** |

### Ví dụ: stream text + audio
```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"Chào bạn, giới thiệu ngắn gọn về trường","stream":true}'
```
Trả về SSE:
```
event: text
data: Chào

event: audio
data: <base64 PCM16 @48kHz>

event: done
data: Chào bạn! Mình là chuyên viên tư vấn...
```

---

## 🎛 Cấu hình nâng cao

### Tốc độ nói (`TTS_SPEED`)
- `1.0` = tự nhiên, `1.2` = nhanh hơn, `0.9` = chậm hơn.
- Map: `max_new_frames = int(300 * TTS_SPEED)`.

### Format text cho TTS
Hàm `format_for_tts()` trong `app/ai.py` tự động:
- Xóa `**markdown**`, `# heading`, emoji, bullet `•`, arrow `→`…
- Thay `…` → `...`, `–` → `-`, `→` → ` đến `.
- Chuẩn hóa khoảng trắng quanh dấu câu `, . ; : ? !`.

### Giọng nói
```python
# Đổi giọng khi gọi
requests.post("/v1/chat/stream", json={"message": "...", "voice": "Ngọc Huyền"})
```

---

## 🐳 Cấu trúc project

```
voice-service/
├── app/
│   ├── main.py        # FastAPI: endpoints + streaming SSE
│   ├── ai.py          # LLM client (OpenAI) + format_for_tts()
│   ├── stt.py         # STT (sherpa-onnx, nghi-stt-v3)
│   ├── tts.py         # TTS (VieNeu) + speed control
│   ├── vad.py         # Silero VAD
│   ├── pipeline.py    # Voice pipeline (VAD→ASR→LLM→TTS, barge-in)
│   └── config.py      # Cấu hình từ env
├── static/
│   └── index.html     # UI voice chat (Web Audio streaming)
├── scripts/           # Tải model
├── models/            # Model STT/TTS/VAD (gitignored)
├── docker-compose.yml
├── Dockerfile         # Multi-stage CPU/GPU
└── requirements.txt
```

---

## 🔧 Troubleshooting

| Lỗi | Nguyên nhân | Xử lý |
|-----|------------|-------|
| Audio nháo nhào | Chunk phát chồng nhau | Đã fix: player schedule nối tiếp |
| Chậm | Gọi LLM 2 lần | Đã fix: 1 LLM call (SSE) |
| Text đọc sai dấu | Markdown/emoji | `format_for_tts()` tự xử lý |
| GPU không nhận | Docker VMM (không phải WSL2) | Chuyển Docker Desktop sang WSL2 backend |
| TTS lỗi CUDA | Driver quá cũ | Dùng image CUDA 12.4 (cu124) |

---

## 📄 License

Dự án nội bộ — vui lòng không phân phối.
