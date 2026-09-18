# 作業分析：財經影音自動化改寫流程 —— 怎麼做、為什麼

> 針對 `assignment-brief.md` 的解讀、素材盤點、技術選型與成本策略。
> 這是「動手前的分析」，不是實作說明；實作完成後 README 另寫。

---

## 1. 題目在考什麼（讀 brief 的結論）

brief 明講三件事，優先順序如下：

| 排序 | 評分重點 | brief 原話 | 對實作的意義 |
|---|---|---|---|
| 1 | **成本意識** | 「本題最重視的不是最終短影音的精緻度」 | 每一步都要有「先算再送、能不送就不送、送過不重送」的機制，而且要**寫出來** |
| 2 | **可以在評審機器上跑起來** | 「步驟需清楚，我會實際測試執行」 | 一條指令能跑完 demo；依賴要少、跨平台、用免費額度 |
| 3 | **合法改寫** | 「不能是原文重製」+ 圖表自製 + 註明出處 | 要有**自動化**的反抄襲檢查與出處注入，不是靠人眼 |

影片精緻度排最後。所以 Step 4「腳本生成短影音」**不要**用昂貴的 AI 影片生成 API 當預設，用程式化合成（TTS + 自製圖表 + 字幕）就夠，並把貴的方案留成可選項來對比成本。這本身就是成本判斷的展示。

---

## 2. 素材盤點（已實際查過）

| 項目 | 事實 | 影響 |
|---|---|---|
| 影片 | `KjAI9r8tnOs`，TVBS《健康2.0》頻道，標題「減輕利息推高房價！房市洗牌逐漸台北化！【精華版】」 | 出處固定寫「根據 TVBS《健康2.0》節目報導指出」 |
| 長度 | **15 分 50 秒**（950 秒） | 逐字稿費用天花板很低，但仍要示範前處理 |
| 語言 | 繁體中文口語，多人談話（主持人 + 來賓） | Whisper 系列常吐簡體 → 需要 OpenCC 轉繁；談話節目有大量閒聊，不是每段都值得做 |
| 字幕 | **無官方字幕、無自動字幕** | 不能走「直接抓字幕、跳過 STT」的捷徑，Step 2 必須真的做 |
| 音軌 | 純音訊格式可直接下載，m4a 48kbps 約 5.8 MB | Step 1 不需要下載影片再抽音，直接 `bestaudio` 就是最省的做法 |
| 內容型態 | 精華版，房市/利率/房價數據導向 | 有數據可畫圖，符合「圖表自製」要求 |

---

## 3. 整體架構

```
 yt-dlp ──► ffmpeg 前處理 ──► STT ──► LLM 篩選+改寫 ──► 合成短影音
 (audio)   (16k mono/去靜音)  (逐字稿)  (兩階段)          (TTS+圖表+字幕)
    │            │              │          │                  │
    └────────────┴──────────────┴──────────┴──────────────────┘
                  data/<video_id>/manifest.json  ← 每一步的快取與成本帳本
```

設計原則：

1. **每一步是獨立、可重跑的 stage**，輸入輸出都是檔案（`data/<video_id>/01_audio.m4a`、`02_transcript.json`、…）。
2. **manifest.json 做冪等**：key = `video_id + stage + 該 stage 設定的 hash`。已存在且 hash 相同就跳過，`--force` 才重做。這直接回答 brief 的「避免同一支影片重複處理」。
3. **先估再跑**：`--dry-run` 只印估算成本不打 API；`MAX_BUDGET_USD` 超過就停。
4. **成本帳本**：每次 API 呼叫把 usage（秒數、tokens、字元數）和估算金額寫進 manifest，最後印總表。評審看得到「這次 demo 花了多少」。

---

## 4. 各步驟選型與成本

### Step 1：影片 → 音檔（成本 $0）

- **工具**：`yt-dlp -f bestaudio`（只拉音軌，不下影片）+ `ffmpeg`。
- **前處理**（這是 brief 點名的「長影音前處理」）：
  - 轉 16 kHz 單聲道 WAV/FLAC → STT 模型本來就只吃這個，檔案縮 2–3 倍。
  - `silenceremove` 去掉靜音與長停頓 → 談話節目通常可省 5–15% 秒數。
  - 可選裁頭尾（片頭曲、結尾 CTA）。
  - 可選 1.1–1.2 倍速（`atempo`）→ Whisper 對中文在 1.2x 內準確率幾乎不掉，秒數再省 15%。**預設關閉**，文件裡列出做過的取捨。
- **風險**：Windows 上 ffmpeg 要另裝（README 寫 `winget install ffmpeg`）；YouTube 偶爾擋 yt-dlp，README 附 `--cookies-from-browser` 備案。

### Step 2：音檔 → 逐字稿（demo 成本 ≈ $0.05–0.10，或 $0）

| 方案 | 單價 | 本片 (≈14 min 去靜音後) | 備註 |
|---|---|---|---|
| **OpenAI `gpt-4o-mini-transcribe`** | $0.003/min | ≈ $0.04 | 品質好、有時間戳、預設建議 |
| OpenAI `whisper-1` | $0.006/min | ≈ $0.09 | 老牌，中文可 |
| Groq `whisper-large-v3-turbo` | 免費額度 | $0 | 快，但免費額度會變 |
| 本機 `faster-whisper` | $0 | $0（CPU 約 5–10 min） | 離線備案；首跑要下模型 |

- **建議**：預設走一個雲端 API（一個 key 就能跑），`--stt local` 切本機當零成本備案。
- **必做後處理**：`opencc` 簡→繁；把逐字稿存成帶時間戳的 segments JSON（後面篩選要用）。
- **分段送**：若音檔 > 25 MB 或 > 10 分鐘，依 VAD 切段再送，避免單次上限，也讓失敗重試只重送壞掉那段。

### Step 3：逐字稿 → 多段短影音腳本（demo 成本 < $0.05）

這是 brief 說「通常最昂貴的步驟」之一，關鍵是 **先篩後寫，兩階段用不同價位的模型**：

- **Pass A（便宜模型，例如 Gemini Flash / GPT-4o-mini / Claude Haiku）**
  輸入整份逐字稿，輸出結構化 JSON：切成 6–10 個主題段落，每段抽出「核心論點、可引用的數據、是否有圖表素材、hook 分數 1–5」。
  16 分鐘中文談話 ≈ 6–8k tokens，一次呼叫，幾乎免費。
- **篩選 gate（純程式，$0）**
  依 hook 分數 ≥ 閾值、有數據可畫圖、不與已選段落重複主題，取 **top-K（預設 K=3）**。其餘段落只留在 JSON 裡，不進下一步。
- **Pass B（較強模型，只對 top-K）**
  每段產一份腳本 JSON：`title / hook / 台詞（含「根據 TVBS《健康2.0》報導指出…」）/ chart_data（數據本身）/ 預估秒數`。
- **反抄襲自動檢查（$0）**
  用逐字稿做 n-gram 索引，算腳本每句與原稿的最長重疊字串與 8-gram 重疊率；超標就退回 Pass B 重寫一次（最多 1 次），仍超標就標記並跳過生成。這讓「改寫非照抄」變成可驗證的閘，不是靠承諾。

### Step 4：腳本 → 短影音（預設 ≈ $0；可選昂貴方案）

| 方案 | 每支 30 秒直式影片 | 說明 |
|---|---|---|
| **程式化合成（預設）** | **$0 – $0.01** | TTS（`edge-tts` 免費，或 OpenAI TTS ≈ $0.003）+ `matplotlib` 畫圖表 PNG + `ffmpeg`/`moviepy` 疊字幕與背景 |
| AI 影片生成（可選 `--renderer ai`） | $1.5 – $15 | Runway / Kling / Veo 類 API，每秒 $0.05–0.5；對財經數據型內容附加價值低 |

- 預設 1080×1920 直式、字幕燒錄、Noto Sans CJK 字型、圖表用 `matplotlib`（符合「可程式化重製圖表」）。
- **成本護欄**：只對通過 Step 3 gate 的段落生成；`--max-clips` 上限；先產 storyboard（圖 + TTS）預覽，確認才 render 完整影片（`--preview-only`）。
- 昂貴方案保留成 flag 而不是拿掉，README 用同一份腳本比較兩者成本，就是題目要的「成本判斷」。

---

## 5. 成本總表（本 demo 一次完整跑）

| 步驟 | 預設方案 | 估算 |
|---|---|---|
| 1. 下載 + 前處理 | yt-dlp + ffmpeg | $0 |
| 2. STT | gpt-4o-mini-transcribe，≈14 min | ≈ $0.04 |
| 3. 篩選 + 改寫 | 便宜模型 1 次 + 強模型 3 次 | ≈ $0.02–0.05 |
| 4. 合成 3 支 | edge-tts + matplotlib + ffmpeg | $0 |
| **合計** | | **≈ $0.06–0.10 / 支長影片** |
| 對照：不做任何篩選、全段用 AI 影片生成 | 8 段 × $5 | ≈ $40+ |

第二次跑同一支影片：**$0**（manifest 全部命中）。這個對照表要放進 README。

---

## 6. Repo 結構草案

```
Quick_assignment/
├── README.md              # 安裝、一鍵跑、成本說明
├── docs/ANALYSIS.md       # 本文件
├── .env.example           # OPENAI_API_KEY=請在此填入你的 key
├── pyproject.toml         # 依賴：yt-dlp, openai, opencc-python-reimplemented, matplotlib, edge-tts, moviepy/ffmpeg-python, pydantic, typer
├── pipeline/
│   ├── cli.py             # `python -m pipeline run --url ... [--dry-run] [--force] [--max-clips 3]`
│   ├── manifest.py        # 冪等快取 + 成本帳本
│   ├── s1_download.py
│   ├── s2_transcribe.py
│   ├── s3_script.py       # 兩階段 + gate + 反抄檢查
│   ├── s4_render.py
│   └── pricing.py         # 單價表，估算用
├── data/<video_id>/       # 中間產物（gitignore，但保留一份 demo 輸出的 transcript/scripts JSON 當證據）
└── tests/                 # 反抄檢查、manifest 冪等的單元測試（不打 API）
```

README 必含：三步安裝（Python 3.12、ffmpeg、`pip install -e .`）、複製 `.env.example`、一條指令跑 demo、`--dry-run` 示範、成本表、每個成本決策的「為什麼」。

---

## 7. 風險與取捨

- **YouTube 反爬**：yt-dlp 偶爾需要更新或 cookies；repo 內附一份已下載的音檔或逐字稿 JSON 當 fallback，讓評審就算下載失敗也能跑 Step 2–4。
- **簡繁**：Whisper 中文常出簡體，OpenCC 必做，README 要說明。
- **免費額度變動**：Groq/Gemini 免費額度可能調整，預設走 OpenAI 小額付費最穩，README 列替代。
- **中文 TTS 品質**：edge-tts 的 `zh-TW-HsiaoChenNeural` 免費且自然，夠 demo 用。
- **Windows**：ffmpeg 路徑、CJK 字型路徑要在 README 交代；主要開發在 Windows 反而能確保評審跨平台不踩雷。
- **合法性邊界**：腳本用自己的話 + 數據 + 出處；不用原影片任何畫面、聲音；圖表用數據重繪。這三點在 README 單獨列一節。

---

## 8. 時程（1–2 天）

| 階段 | 內容 | 預估 |
|---|---|---|
| D1 上午 | 骨架、manifest、Step 1–2 跑通、成本帳本 | 3 h |
| D1 下午 | Step 3 兩階段 + gate + 反抄檢查 + 測試 | 3 h |
| D2 上午 | Step 4 程式化合成、跑出 3 支 demo | 3 h |
| D2 下午 | README、成本文件、清 key、push、對照 brief 逐條核對 | 2 h |

---

## 9. 開工前需要你決定的事

1. **手上有哪些 API key / 免費額度**：OpenAI、Gemini、Groq、Anthropic？決定 Step 2/3 預設走誰。
2. **Step 4 要不要真的接一個 AI 影片生成 API 當可選項**（會花幾美元），還是只在文件裡估算對比？
3. **Repo public 還是 private**（private 要加 kay.kw.yu@rgxtechnology.com）。
4. 腳本語言預設繁中；短影音尺寸預設 1080×1920 直式。沒意見就照這個做。
