# finvid-rewriter：財經影音 → 原創短影音的成本導向自動化流程

把一支 YouTube 財經節目（[TVBS《健康2.0》房市精華版](https://www.youtube.com/watch?v=KjAI9r8tnOs)，15 分 50 秒）
自動轉成數支 30–45 秒、直式、**用自己的話改寫並註明出處**的短影音。

四個步驟都在 `pipeline/stages/`，每一步都先估算、再判斷要不要送、送過就不重送：

| 步驟 | 做什麼 | 工具 | 本片實際花費（帳本） |
|---|---|---|---|
| 1 `s1_download` | 只拉音軌，轉 16 kHz 單聲道，去靜音 | yt-dlp + ffmpeg | $0 |
| 2 `s2_transcribe` | 分段送 STT，簡轉繁 | OpenAI `gpt-4o-mini-transcribe`（或本機 faster-whisper） | $0.047（15.8 分鐘） |
| 3 `s3_script` | 便宜模型切段打分 → 篩選 → 只對前 3 段用強模型寫腳本 → 反抄襲閘 | OpenAI `gpt-5-mini` + `gpt-5` | $0.061（8 段候選，3 段寫腳本，省下 5 次強模型呼叫） |
| 4 `s4_render` | 免費 TTS + matplotlib 重繪圖表 + ffmpeg 合成字幕 | edge-tts + matplotlib + PIL + ffmpeg | $0（3 支共 118 秒） |
| | | **合計** | **$0.108** |

對照：同樣 3 支用 AI 影片 API 生成約 $29.5。**第二次跑同一支影片：$0**（manifest 快取全命中）。
數字來自 [data/demo/manifest.json](data/demo/manifest.json)，clone 下來不用 key 就能用 `finvid costs --url demo` 重印。詳細成本假設與決策見 [docs/COST.md](docs/COST.md)，設計分析見 [docs/ANALYSIS.md](docs/ANALYSIS.md)。

---

## 1. 在本機跑起來

需求：Python 3.11+、ffmpeg、一把 OpenAI API key。整支 demo 約 $0.1，帳戶有幾塊美金餘額（或新帳號的試用額度）就夠；
沒有 key 也能做三件事：跑測試、`finvid run --dry-run` 看估算、`finvid serve` 看 repo 內附的 demo 產出。
第 2 步可改 `FINVID_STT_PROVIDER=local`（本機 faster-whisper，$0）；第 3 步的 LLM 目前只接 OpenAI。

```bash
# 1. 取得程式碼
git clone https://github.com/markl-a/finvid-rewriter.git && cd finvid-rewriter

# 2. 裝 ffmpeg（已裝可跳過）
winget install Gyan.FFmpeg        # Windows
brew install ffmpeg               # macOS
sudo apt install ffmpeg           # Ubuntu

# 3. 建虛擬環境並安裝
python -m venv .venv              # Windows 若 python 沒反應（Store 別名），改用 py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows；macOS/Linux 用 source .venv/bin/activate
#   PowerShell 被 ExecutionPolicy 擋住時：Set-ExecutionPolicy -Scope Process Bypass，或不啟用、直接用 .venv\Scripts\finvid.exe
pip install -e ".[dev]"

# 4. 填 key
copy .env.example .env            # macOS/Linux 用 cp
# 打開 .env，把 OPENAI_API_KEY 填進去
```

### 先看估算（不打付費 API、不下載音訊）

```bash
finvid run --dry-run
```

會印出每一步的預估費用與總額。dry-run 唯一的網路動作是用 yt-dlp 向 YouTube 抓影片長度與標題（免費），不會建立 `data/<video_id>/`。預設 `FINVID_MAX_BUDGET_USD=1.00`，任何一次呼叫會讓總額超過就直接中止。

### 真的跑

```bash
finvid run
```

等同 `finvid run --url https://www.youtube.com/watch?v=KjAI9r8tnOs --max-clips 3`。產出在 `data/KjAI9r8tnOs/`：

```
01_audio.wav          前處理後的音檔
01_info.json          標題、頻道、原始/處理後長度
02_transcript.json    帶時間戳的繁中逐字稿
03_scripts.json       所有候選段落（含被跳過的原因）+ 通過的腳本 + 被退回的腳本
04_clips/clip_XX.mp4  短影音（1080×1920）
04_clips/chart_XX.png 用數據重繪的圖表
04_render.json
manifest.json         每一步的快取 key + 成本帳本 + 每次執行紀錄
```

再跑一次 `finvid run`：四個 stage 全部 cache hit，花費 $0。要重做用 `--force`，或 `finvid clean --stage s3` 只重做第 3 步以後。

### Demo 產出（不用跑也看得到）

[data/demo/](data/demo/) 是本片實際跑出來的結果：逐字稿、8 段候選與每段的選中或跳過原因、3 支腳本（含反抄襲檢查數字）、3 支 mp4、3 張重繪的圖表、完整成本帳本。

| 段落 | hook 分數 | 有數據 | 決定 |
|---|---|---|---|
| 1 臺北購屋門檻與貸款負擔 | 4 | 是 | 跳過：與段落 4 數字相同 |
| 2 中古屋年限與屋況取捨 | 2 | 否 | 跳過：分數低於門檻 3 |
| 3 購屋者的個人規劃與壓力 | 3 | 否 | 跳過：與段落 4 數字相同 |
| 4 新青安方案的月付與年限比較 | 5 | 是 | **選中** → 「新青安月付差多大？」 |
| 5 新青安的制度性疑慮 | 4 | 是 | **選中** → 「新青安隱憂一次看」 |
| 6 臺南安平區與個案價格 | 4 | 是 | 跳過：超過 max_clips |
| 7 臺中房價飆升與高價盤整區 | 4 | 是 | 跳過：超過 max_clips |
| 8 市場回檔與房貸負擔率 | 5 | 是 | **選中** → 「回檔與房貸壓力」 |

三支腳本的反抄襲檢查：6 字 n-gram 重疊率都是 0%，最長共同子字串 3–5 字（都是數字或專有名詞）。

### 本機 web 介面（把成本決策攤開來看）

```bash
finvid serve
```

開 http://127.0.0.1:8000 ：可以貼網址執行、看即時 log、每個 stage 的花費與快取狀態、哪些段落被選中／跳過及原因、實際花費 vs 「不篩選＋用 AI 影片 API」的對照、直接預覽產出的影片。

**沒有 key 也能看**：clone 下來直接 `finvid serve`，左邊「已處理影片」會列出 `demo`（repo 內附的實跑結果），點進去就是完整儀表板與三支影片。

### 其他指令

```bash
finvid run --until s2         # 只跑到逐字稿
finvid run --max-clips 1      # 只產 1 支
finvid costs                  # 印本片（KjAI9r8tnOs）的成本帳本
finvid costs --url demo       # 印 repo 內附 demo 的帳本（不需 key）
python -m pytest -q           # 單元測試（不需要 key、不需要網路）
```

---

## 2. 成本意識：這個流程在哪裡省錢

brief 點名的三件事，對應的機制：

**「音檔轉文字時，是否對長影音做前處理」**
- 只下載音軌（`bestaudio`），不下載影片。
- 轉 16 kHz 單聲道 PCM，這是 STT 模型的原生輸入。
- ffmpeg `silenceremove` 去掉 0.7 秒以上、低於 −35 dB 的停頓。**誠實說明：本片只省 0.2%**，因為節目全程有背景音樂床（平均 −22 dB），沒有真正的靜音。這個機制對純談話或 podcast 類素材才有 5–15% 的效果，對這支影片幾乎無效，我保留它但不誇大。
- 對本片真正有效的槓桿是 `FINVID_SPEEDUP=1.2`（950 秒 → 792 秒，STT 費用 −17%）和 `FINVID_TRIM_HEAD_SEC` / `TRIM_TAIL_SEC` 裁掉片頭片尾。兩者預設關閉：加速會讓中文辨識多一些錯字，裁頭尾要先看過影片才知道秒數，都不適合當無人值守的預設值。

**「拆腳本、生成短影音是否先篩選再送」**
- Pass A 用便宜模型（`gpt-5-mini`）讀整份逐字稿一次，切成 5–10 段並給每段 hook 分數、有無可畫圖的數據。
- 篩選閘是純程式：分數低於門檻、主題重複、超過 `--max-clips` 的段落都不進下一步。候選段落通常 6–10 段，只有 3 段送強模型，其餘在 `03_scripts.json` 留下跳過原因。
- 強模型（`gpt-5`）每段一次呼叫，是流程中單價最高的地方，所以只在這裡用。
- 生成影片不用 AI 影片 API：TTS 免費、圖表用 matplotlib、合成用 ffmpeg。帳本裡仍記一筆「若用 Runway/Kling 類 API 會花多少」當對照（3 支 35 秒約 $26）。

**「是否避免同一支影片重複處理」**
- `data/<video_id>/manifest.json` 記每個 stage 的設定 hash 與輸出檔。設定沒變、檔案還在就跳過。
- 改了第 2 步的設定，第 3、4 步自動失效重做，第 1 步不動。
- 同一支影片同時只允許一個執行：`data/<video_id>/.running.lock`（跨程序，CLI 與 web UI 共用；兩個終端同時 `finvid run` 第二個會被擋下，exit 4），逾 3 小時的殘留鎖視為當機遺留自動接手。

**額外護欄**
- `--dry-run`：全流程只估算，不呼叫付費 API、不下載音訊、不寫任何檔案（只向 YouTube 抓 metadata）。
- `FINVID_MAX_BUDGET_USD`：每次 API 呼叫前先算預估費用，超過就在呼叫前中止。
- 每筆 API 用量（秒數、tokens、字元）和單價都寫進 manifest，`finvid costs` 或 web UI 可查。

---

## 3. 版權與合法改寫

- **改寫非照抄**：Pass B 的提示詞明確要求重組句構、換用詞、數字不變。之後用程式檢查：腳本的 6 字元 n-gram 有多少比例出現在原逐字稿（門檻 15%）、最長共同子字串（門檻 12 字）。超標退回重寫一次，仍超標就丟到 `rejected_clips`，不進生成。
- **圖表自製**：腳本只帶數據（`chart.series[].points`），圖表由 matplotlib 從數據畫，不碰原影片任何畫面。
- **註明出處**：「根據 TVBS《健康2.0》報導指出」由程式模板注入每支腳本的第 1 或 2 句，不靠 LLM 記得寫；影片底部固定顯示「資料來源」字卡。
- 不使用原影片的畫面、聲音；產出只包含改寫後的文字、自製圖表、合成語音。

---

## 4. 選型理由

| 步驟 | 選擇 | 為什麼 | 替代 |
|---|---|---|---|
| 下載 | yt-dlp | 免費、可只拉音軌、跨平台 | pytube（維護較差） |
| STT | `gpt-4o-mini-transcribe` $0.003/min | 最便宜的雲端選項，中文品質足夠，一把 key 就能跑全流程 | `FINVID_STT_PROVIDER=local` 用 faster-whisper，$0 但要下載模型、CPU 跑幾分鐘 |
| 切段打分 | `gpt-5-mini` | 整份逐字稿只讀一次，任務是結構化摘要，不需要強模型 | `gpt-5-nano` 更便宜但 JSON 穩定度較差 |
| 寫腳本 | `gpt-5`（reasoning low） | 改寫品質直接決定合法性與可看性，值得花；只對篩過的段落用 | `gpt-4.1-mini` 便宜 5 倍，品質可接受 |
| TTS | edge-tts `zh-TW-HsiaoChenNeural` | 免費、台灣腔、自然度夠 demo | OpenAI `gpt-4o-mini-tts` 約 $0.01/支 |
| 圖表 | matplotlib | 題目要求可程式化，CJK 字型可控 | Plotly（要 kaleido 輸出圖片，依賴較重） |
| 合成 | PIL + ffmpeg | 零依賴問題，Windows/macOS 都穩；字幕用 PNG overlay 避開 ffmpeg subtitles filter 在 Windows 的路徑地雷 | moviepy（慢、依賴多） |
| UI | FastAPI + 單一 HTML | 不用 build、跨平台、不需部署；桌面 app 會被 Gatekeeper/防毒擋，雲端網站會曝露 key | Streamlit |

---

## 5. 專案結構

```
pipeline/
  cli.py            finvid run / serve / costs / clean
  config.py         所有影響成本的設定（.env）
  context.py        RunContext、run_stage（快取、dry-run、預算閘）
  manifest.py       manifest.json：冪等快取 + 成本帳本
  pricing.py        單價表（唯一寫死數字的地方）
  llm.py            LLM 呼叫封裝，回 JSON + 成本
  plagiarism.py     n-gram 重疊 / 最長共同子字串
  stages/           s1_download s2_transcribe s3_script s4_render
  render/           tts chart compose fonts
  ui/               server.py + static/index.html
tests/              不打 API、不需網路的單元測試
docs/               ANALYSIS.md（設計分析）COST.md（成本假設與決策）
data/<video_id>/    產出與 manifest（gitignore；repo 內保留一份 demo 的 JSON 產出）
```

---

## 6. 已知限制

- YouTube 偶爾擋 yt-dlp。失敗時先 `pip install -U yt-dlp`；仍不行可把音檔手動放到 `data/<video_id>/01_audio.wav`。
- `gpt-4o-mini-transcribe` 不回傳片段時間戳，中文輸出也沒有標點、只用空格分句。逐字稿依空格切成約 30 字的段落，時間是依每個 600 秒分段內的字數線性內插；切換到 `whisper-1` 可得到精確時間戳（貴一倍）。
- edge-tts 需要網路；CJK 字型在 Windows 用微軟正黑體、macOS 用 PingFang，Linux 需自行安裝 Noto Sans CJK 並設 `FINVID_FONT`。
- 單價表是 2026-09-18 查的，變動請改 `pipeline/pricing.py`。
