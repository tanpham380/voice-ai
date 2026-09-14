# Voice AI Service — STT → LLM → TTS (streaming)

Dịch vụ giọng nói tiếng Việt chạy độc lập: **thu âm → nhận dạng (STT) → AI tư vấn (LLM) → đọc lại (TTS)**,
với khả năng **streaming thời gian thực** (text và audio phát đồng bộ từng câu).

```
mic / text ──► STT (nghi-stt-v3, sherpa-onnx)
              ──► LLM (OpenAI-compatible: DeepSeek qua AI-Box gateway)
                    ──► TTS (ZeroTTS zero-shot, 48 kHz, CPU)
                          ──► Web Audio (phát stream tiếp nối)
```

Công nghệ: **Python + FastAPI**, chạy trên **CPU** (ONNX, không cần PyTorch/GPU).

---

## ✨ Tính năng

- **Streaming đồng bộ (1 LLM call)**: text và audio được stream cùng lúc qua SSE
  (`event: text` + `event: audio`), không gọi LLM 2 lần → không lặp, không chậm gấp đôi.
- **STT tiếng Việt**: `nghi-stt-v3` (Zipformer transducer, sherpa-onnx), chạy offline.
- **LLM tư vấn**: OpenAI-compatible (mặc định DeepSeek qua AI-Box gateway),
  hỗ trợ streaming token, tắt "thinking" để phản hồi nhanh.
- **TTS ZeroTTS zero-shot**: 48 kHz, 8 giọng tiếng Việt preset, chạy trên **CPU** (ONNX, không cần PyTorch). TTFA ~70 ms, không khoảng lặng chết giữa các câu.
- **Format text trước khi TTS**: tự động loại bỏ markdown, emoji, ký tự đặc biệt,
  chuẩn hóa dấu câu tiếng Việt → giọng đọc tự nhiên, rõ ràng.
- **Giọng nói**: chọn qua tham số `voice` (mặc định `maichi`). 8 preset: maichi, baotrang, kimoanh, hamy, giahuy, huuduc, quangminh, tiendat.
- **Web Audio player**: phát PCM16 stream nối tiếp (không chồng chéo → không nháo nhào).
- **Barge-in**: người dùng có thể cắt ngang khi bot đang nói (pipeline mode).

---

## 🚀 Nhanh chóng bắt đầu

### 1. Yêu cầu
- Docker Desktop với **WSL2 backend** + NVIDIA GPU (driver hỗ trợ GPU-PV).
- Model STT/TTS (tải 1 lần):

```bash
python scripts/download_models.py        # STT (nghi-stt-v3) + Silero VAD
python scripts/download_tts_models.py    # ZeroTTS weights (~900 MB, tải 1 lần)
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
| `TTS_DEFAULT_VOICE` | Giọng mặc định | `maichi` |
| `TTS_CFG_SCALE` | Cường độ định hình giọng | `1.0` |
| `TTS_AUDIO_TEMPERATURE` | Nhiệt độ lấy mẫu audio | `0.8` |

### 3. Build & chạy (CPU)
```bash
docker compose build
docker compose up -d
```
Mở **http://localhost:8000** để test voice chat.

> ZeroTTS chạy CPU-only (ONNX). Không cần GPU NVIDIA.

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

### Tốc độ nói (`TTS_MIN_FRAMES` / `TTS_MAX_FRAMES`)
- ZeroTTS điều khiển độ dài bằng ngân sách frame, không phải nhân tốc độ.
- `TTS_MIN_FRAMES` (mặc định `4`) = độ dài tối thiểu; `TTS_MAX_FRAMES` (mặc định `1500`).
- Giảm `TTS_MIN_FRAMES` để câu ngắn/cắn gọn hơn.

### Format text cho TTS
Hàm `format_for_tts()` trong `app/ai.py` tự động:
- Xóa `**markdown**`, `# heading`, emoji, bullet `•`, arrow `→`…
- Thay `…` → `...`, `–` → `-`, `→` → ` đến `.
- Chuẩn hóa khoảng trắng quanh dấu câu `, . ; : ? !`.

### Giọng nói
```python
# Đổi giọng khi gọi
requests.post("/v1/chat/stream", json={"message": "...", "voice": "maichi"})
```

---

## 🐳 Cấu trúc project

```
voice-service/
├── app/
│   ├── main.py        # FastAPI: endpoints + streaming SSE
│   ├── ai.py          # LLM client (OpenAI) + format_for_tts()
│   ├── stt.py         # STT (sherpa-onnx, nghi-stt-v3)
│   ├── tts.py         # TTS (ZeroTTS zero-shot)
│   ├── vad.py         # Silero VAD
│   ├── pipeline.py    # Voice pipeline (VAD→ASR→LLM→TTS, barge-in)
│   └── config.py      # Cấu hình từ env
├── static/
│   └── index.html     # UI voice chat (Web Audio streaming)
├── scripts/           # Tải model
├── models/            # Model STT/TTS/VAD (gitignored)
├── docker-compose.yml
├── Dockerfile         # Multi-stage CPU
└── requirements.txt
```

---

## 🔧 Troubleshooting

| Lỗi | Nguyên nhân | Xử lý |
|-----|------------|-------|
| Audio nháo nhào | Chunk phát chồng nhau | Đã fix: player schedule nối tiếp |
| Chậm | Gọi LLM 2 lần | Đã fix: 1 LLM call (SSE) |
| Text đọc sai dấu | Markdown/emoji | `format_for_tts()` tự xử lý |
| TTS chậm / nóng CPU | Tăng `intra_op_num_threads` hoặc giảm `TTS_MIN_FRAMES` | Chỉnh `config.py` / `.env` |
| Voice không phù hợp | Đổi `voice` về preset khác | `maichi` / `giahuy` |

---

## 📄 License

Dự án nội bộ — vui lòng không phân phối.
