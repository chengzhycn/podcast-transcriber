# podcast-transcriber

将[小宇宙](https://www.xiaoyuzhoufm.com/)（xiaoyuzhoufm.com）播客单集自动转换为文字稿，并可选用 LLM 整理成可发布的中文 blog 文章。

```
小宇宙单集 URL → 抓取音频 → 语音转文字（STT）→（可选）LLM 整理 → Markdown
```

## 功能

- **搜索**：通过 DuckDuckGo `site:xiaoyuzhoufm.com` 搜索播客/单集，无需登录或 token
- **抓取**：从小宇宙页面 `__NEXT_DATA__` 中提取单集元数据和音频 CDN 直链
- **转录**：支持两种 STT 后端
  - `local`（默认）：本地 [FunASR](https://github.com/modelscope/FunASR) Docker 服务，paraformer-zh + cam++，中文准确率高，自带说话人分离（`[SPK0]` / `[SPK1]` ...）
  - `openai`：OpenAI Whisper API
- **整理**：调用 OpenAI 兼容的 Chat Completion API（默认 `qwen3.7-max`），把转录稿整理成有标题、分节、引用、点评、延伸阅读的 blog 文章

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
cp .env.example .env   # 然后编辑 .env 填入你的配置
```

### 2.（推荐）启动本地 FunASR 转录服务

```bash
docker compose -f docker-compose.funasr.yml up -d
# 首次启动会自动下载模型（约 5-10 分钟），用下面命令检查就绪状态：
curl http://localhost:18902/health
```

> 模型权重持久化在 Docker volume 中，重启容器无需重新下载。
> 4 个模型（paraformer + vad + punc + cam++）同时加载约需 5GB 内存，请确保 Docker Desktop 的虚拟机内存上限（Settings → Resources → Advanced）不低于 6-7GB。

### 3. 搜索想要转换的播客/单集

```bash
python3 main.py search-cmd "得意忘形"
```

### 4. 运行完整流程

```bash
# 转录 + LLM 整理成 blog（默认）
python3 main.py run "https://www.xiaoyuzhoufm.com/episode/xxxx"

# 只要转录稿，不调用 LLM
python3 main.py run "https://www.xiaoyuzhoufm.com/episode/xxxx" --transcript-only
```

## CLI 参考

```
python3 main.py search-cmd <关键词> [--limit N]
python3 main.py run <episode-url> [OPTIONS]
```

| 选项 | 说明 |
|------|------|
| `--stt local\|openai` | STT 后端，默认读取 `.env` 中 `STT_PROVIDER`（默认 `local`） |
| `--transcript PATH` | 已有转录稿文件，跳过 STT 直接整理 |
| `--audio-only` | 仅下载音频，不转录 |
| `--transcript-only` | 转录到文字稿即停止，不调用 LLM 整理 |
| `--llm-model TEXT` | 整理用的 LLM 模型，默认读取 `.env` 中 `LLM_MODEL` |
| `--output PATH` | 指定输出 Markdown 路径 |

输出文件保存在 `OUTPUT_DIR`（默认 `./output/`）：

```
output/
├── audio/<播客名>_<集名>_<日期>.<ext>          # 下载的音频
├── <播客名>_<集名>_<日期>_transcript.txt        # 转录稿
└── <播客名>_<集名>_<日期>_blog.md               # 整理后的 blog（未加 --transcript-only 时生成）
```

## 配置（.env）

参见 [`.env.example`](.env.example)，关键项：

| 变量 | 说明 |
|------|------|
| `STT_PROVIDER` | 默认 STT 后端：`local` \| `openai` |
| `ASR_LOCAL_URL` | 本地 FunASR 服务地址，默认 `http://localhost:18902` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | LLM 整理用的 OpenAI 兼容 API 配置 |
| `OUTPUT_DIR` | 输出目录，默认 `./output` |

## 项目结构

```
.
├── main.py              # CLI 入口（typer）
├── searcher.py          # 通过 DuckDuckGo 搜索播客/单集
├── fetcher.py           # 抓取单集元数据 + 下载音频
├── transcriber.py       # 两种 STT 后端的调用封装
├── organizer.py         # 调用 LLM 整理转录稿成 blog
├── config.py            # 读取 .env 配置
├── funasr_server/       # 本地 FunASR 转录服务（FastAPI + OpenAI 兼容 API）
│   ├── server.py
│   └── Dockerfile
└── docker-compose.funasr.yml
```

## 本地 FunASR 服务

`funasr_server/` 是一个独立的 FastAPI 服务，把 [FunASR](https://github.com/modelscope/FunASR) 的 `paraformer-zh + fsmn-vad + ct-punc + cam++` 模型组合包装成 OpenAI 兼容的 `POST /v1/audio/transcriptions` 接口，可单独部署使用：

```bash
docker compose -f docker-compose.funasr.yml up -d
curl http://localhost:18902/health
# {"status":"ok","model":"paraformer-zh+cam++"}
```

返回结果带说话人标签：

```
[SPK0] 你这节目开头一般是...
[SPK1] 但我也很无所谓了...
```

性能参考（Apple Silicon Mac，Docker linux/arm64，CPU 推理）：约 2.5 小时音频 ≈ 15 分钟转录完成（rtf ≈ 0.09）。

## 故障排查

- **FunASR 容器反复重启 / 内存不足**：检查 `docker-compose.funasr.yml` 的 `mem_limit`，以及 Docker Desktop 虚拟机的内存上限（Settings → Resources → Advanced）
- **搜索无结果**：DuckDuckGo 偶尔触发反爬验证，换个关键词重试，或直接从小宇宙 App 分享单集链接
- **LLM 调用报错**：检查 `.env` 中 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` 是否正确
