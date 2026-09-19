# finvid-rewriter：財經影音 → 原創短影音的成本導向自動化流程

把一支 YouTube 財經節目（[TVBS《健康2.0》房市精華版](https://www.youtube.com/watch?v=KjAI9r8tnOs)，15 分 50 秒）
自動轉成數支 30–45 秒、直式、**用自己的話改寫並註明出處**的短影音。

四個步驟都在 `pipeline/stages/`，每一步都先估算、再判斷要不要送、送過就不重送：

| 步驟 | 做什麼 | 工具 | 本片實際花費（帳本） |
|---|---|---|---|
| 1 `s1_download` | 只拉音軌，轉 16 kHz 單聲道，去靜音 | yt-dlp + ffmpeg | $0 |
| 2 `s2_transcribe` | 分段送 STT，簡轉繁 | OpenAI `gpt-4o-mini-transcribe`（或本機 faster-whisper） | $0.047（15.8 分鐘） |
| 3 `s3_script` | 便宜模型切段打分 → 篩選 → 只對前 3 段用強模型寫腳本 → 反抄襲閘 → 數字溯源閘 | OpenAI `gpt-5-mini` + `gpt-5` | $0.052（Pass B 3 段；Pass A 命中快取，首次另 $0.006。8 段候選只寫 3 段，省下 5 次強模型呼叫） |
| 4 `s4_render` | **AI 生成的開場鏡頭 + 每句一段真實素材**鋪滿整支，圖表卡疊在畫面上，免費 TTS + ffmpeg 合成字幕 | AI：HF ZeroGPU → Pixazo → 本機 ComfyUI（同一個 LTX-Video 模型，免費備援鏈）；素材：Pexels；edge-tts + matplotlib + PIL + ffmpeg | $0（3 支共 110 秒；3 段 AI 鏡頭共 39 秒、23 段 Pexels 素材——首次 23 次搜尋，之後全部快取 0 次） |
| | | **合計** | **$0.099** |

對照：同樣 3 支整支用 AI 影片 API 生成約 $27.5；換成付費的 MiniMax 只做開場鏡頭是 $0.81（3 × $0.27），流程支援但預設不用。**第二次跑同一支影片：$0**（manifest 快取全命中，AI 鏡頭也依 prompt hash 快取）。
數字來自 [data/demo/manifest.json](data/demo/manifest.json)，clone 下來不用 key 就能用 `finvid costs --url demo` 重印。詳細成本假設與決策見 [docs/COST.md](docs/COST.md)，設計分析見 [docs/ANALYSIS.md](docs/ANALYSIS.md)。

### brief 的三個評分點，本片實際省了多少

| brief 問的 | 機制 | 本片的數字（帳本） |
|---|---|---|
| **STT 前是否對長影音做前處理，而非整支直接送** | 只拉音軌不拉影片（15 MB，不是幾百 MB）→ 16 kHz 單聲道 → ffmpeg 去靜音 → 600 秒分段、每段送前過預算閘；可選 `FINVID_SPEEDUP` / 裁頭尾 | 去靜音**只省 0.2%**（950 → 948 秒）：本片全程有音樂床，沒有真靜音，[誠實寫在第 2 節](#2-成本意識這個流程在哪裡省錢)。加速 1.2× 可省 17%，預設關（會增加錯字） |
| **拆腳本、生成影片這些貴的步驟，是否先篩選再送** | 便宜模型 `gpt-5-mini` 讀一次全文打分 → 純程式閘（門檻／去重／上限）→ 只有選中的段落才送 `gpt-5` 寫腳本 → 只有通過反抄襲＋數字溯源的腳本才生成影片，每支固定 1 段 AI 鏡頭；付費 provider 走 `FINVID_MAX_BUDGET_USD` | 8 段候選只寫 3 段，省 5 次強模型呼叫 **≈ $0.088**（Pass B 實付 $0.052）；影片生成只做 3 段 5 秒開場而非 8 段整支：**$0 vs 整支 AI 生成 $27.5**（付費 MiniMax 只做開場也才 $0.81） |
| **是否避免同一支影片重複處理、重複計費** | 五層快取：stage 設定 hash（manifest）→ Pass A 回覆 → 每段 AI 鏡頭的 prompt hash → Pexels 搜尋與下載 → 跨程序 `.running.lock`；上游設定變了才讓下游失效 | 第二次 `finvid run`：**$0、0 次 API 呼叫**。整個開發過程重跑 17 次（提示詞改 8 版、版面改 8 版）累計 **$0.385**，沒有快取會是 17 × $0.11 ≈ **$1.9** |

---

## 1. 在本機跑起來

需求：Python 3.11+、ffmpeg、一把 OpenAI API key。整支 demo 約 $0.1，帳戶有幾塊美金餘額（或新帳號的試用額度）就夠；
**預設 `finvid run` 產出的是靜態卡版本**（不需要 GPU、不需要其他 key）；要重現 [data/demo/](data/demo/) 那種「AI 開場 + 真實素材」的成品，看下面「重現 demo」。
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
03a_pass_a.json       Pass A（便宜模型）的原始回覆快取：Pass B 中途中止或只改 Pass B 提示詞時，不再付一次 Pass A
03_scripts.json       所有候選段落（含被跳過的原因）+ 通過的腳本 + 被退回的腳本
04_clips/clip_XX.mp4  短影音（1080×1920）
04_clips/chart_XX.png 用數據重繪的圖表
04_clips/ai_XX*.mp4   AI 生成的 5 秒鏡頭（FINVID_AI_VIDEO 設定時；.json 是 provider、prompt 與秒數）
data/_broll/          Pexels 實拍素材快取（FINVID_BROLL=pexels 時；跨影片共用，04_render.json 的 broll_paths 以 ../_broll/ 指向這裡）
04_render.json
manifest.json         每一步的快取 key + 成本帳本 + 每次執行紀錄
```

再跑一次 `finvid run`：四個 stage 全部 cache hit，花費 $0。要重做用 `--force`，或 `finvid clean --stage s3` 只重做第 3 步以後。

### 重現 demo（AI 開場鏡頭 + 每句真實素材，$0，兩把免費 key）

```bash
# .env 加兩行：HF_TOKEN=hf_...（huggingface.co → Settings → Access Tokens，免費帳號即可）
#              PEXELS_API_KEY=...（pexels.com/api，註冊即發）
FINVID_AI_VIDEO=hf FINVID_BROLL=pexels finvid run      # PowerShell: $env:FINVID_AI_VIDEO="hf"; $env:FINVID_BROLL="pexels"; finvid run
```

會多做：每支 clip 一段 5 秒 AI 開場鏡頭（HF ZeroGPU，約 20 秒/段、$0）、每句台詞一段 Pexels 直式素材（約 20 次搜尋、$0）。[data/demo/](data/demo/) 就是這條路徑跑出來的（`FINVID_AI_VIDEO=hf,pixazo,comfy` 備援鏈，實際用到的是 `hf`）。沒有 GPU、額度用完、沒 key 都會明確報錯而不是默默退回靜態卡；細節與其他 provider 見「第 4 步的 AI 生成畫面」。

### Demo 產出（不用跑也看得到）

[data/demo/](data/demo/) 是本片實際跑出來的結果：逐字稿、8 段候選與每段的選中或跳過原因、3 支腳本（含反抄襲檢查數字）、3 支 mp4、3 張重繪的圖表、完整成本帳本。

| 段落 | hook 分數 | 有數據 | 決定 |
|---|---|---|---|
| 1 新青安與購屋門檻 | 4 | 是 | 跳過：與段落 5 數字相同 |
| 2 租屋轉買房的生活考量 | 3 | 否 | 跳過：超過 max_clips |
| 3 中古屋年限與空間折衷 | 3 | 是 | 跳過：超過 max_clips |
| 4 個人理財與還款規劃 | 3 | 是 | 跳過：超過 max_clips |
| 5 新青安政策細節與試算 | 5 | 是 | **選中** → 「新青安月付差多大？」 |
| 6 政策風險與個人負擔率 | 4 | 是 | 跳過：與段落 8 數字相同 |
| 7 中南部房市漲勢與單價 | 5 | 是 | **選中** → 「安平台中價帶爆衝」 |
| 8 房市回檔與負擔率台北化 | 5 | 是 | **選中** → 「房市回檔與負擔臺北化」 |

三支腳本的反抄襲檢查：6 字 n-gram 重疊率都是 0%，最長共同子字串 3–5 字（都是數字或專有名詞）；數字溯源閘：10 個圖表點與口白中的每個數字都能在逐字稿找到（`numbers_unverified` 全空）。

### 成品展示

[docs/showcase/](docs/showcase/README.md)：三個免費 AI provider 用同一個 prompt 生成的鏡頭並排比較（影片 + 影格），以及完整成品的影格帶。

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

### 第 4 步的 AI 生成畫面（可選，$0；付費選項受預算護欄）

預設 `FINVID_AI_VIDEO=none`：影片是靜態卡 + 字幕 + 旁白，秒級完成、不需要 GPU 也不需要 key。

| `FINVID_AI_VIDEO` | 跑在哪 | 費用 | 每段 5 秒鏡頭 | 需要什麼 |
|---|---|---|---|---|
| `hf` | Hugging Face ZeroGPU 上 Lightricks 官方的 LTX-Video Space | **$0** | ~25 秒 | 什麼都不用；匿名每天約 1–2 段，填免費帳號的 `HF_TOKEN` 額度較大 |
| `comfy` | 你自己的 GPU（本機 ComfyUI） | $0 | 80–120 秒（Radeon 8060S 內顯） | 裝 ComfyUI + 11.5 GB 模型（下面） |
| `pixazo` | Pixazo 代管的 LTX-Video 端點（預覽期免費方案） | **$0** | ~60 秒 | 免費 key（`PIXAZO_API_KEY`，不用信用卡；每分鐘 60 次請求） |
| `minimax` | MiniMax Hailuo API（付費） | $0.27/段（768P；512P $0.08、1080P $0.54） | 20–60 秒 | 付費 key（`MINIMAX_API_KEY`）；輸出 16:9，合成時置中裁成 9:16 |
| `kling` | Kling API（付費；已實作、尚未實跑） | $0.18/段（kling-v1 std 5 s；v2.5-turbo $0.31） | 1–3 分鐘 | Access Key + Secret Key（`KLING_ACCESS_KEY`/`KLING_SECRET_KEY`，程式自簽 JWT）；原生 9:16 |
| `hf,pixazo,comfy` | 先雲端免費額度，用完自動換本機 | $0 | — | 以上 |

免費額度就是這一步的「預算」：額度用完會明確報錯（或依備援鏈換下一個），已生成的鏡頭都有快取，隔天再跑只補缺的。這跟 OpenAI 那邊的 `FINVID_MAX_BUDGET_USD` 是同一個思路，只是單位從美金變成 GPU 秒。付費的 `minimax` 則直接走美金那條路：每段生成前先以表定價格（`minimax.py` 的 `PRICES_USD`，2026-09-19 看到的價格）過 `FINVID_MAX_BUDGET_USD` 的護欄，跟每一次 OpenAI 呼叫一樣，超預算就在呼叫前中止；備援鏈裡把它放最後，只有免費的都用完才會付錢。

不管哪個 provider，整支 clip 的背景都是 AI 生成的直式影片：每支生成 `FINVID_AI_SHOTS_PER_CLIP`（預設 2）段 5 秒鏡頭——第 1 段用第 3 步 Pass B 寫的 `ai_shot`（開場畫面描述），其餘用該段台詞的 `visual` 關鍵字組 prompt——每段做成正放+倒放的無縫迴圈鋪滿它負責的時間窗；重繪的圖表從第一句正文起以卡片疊在畫面中段，標題／字幕／出處疊最上層。兩個 provider 用的是同一個模型（[LTX-Video 2B 蒸餾版](https://huggingface.co/Lightricks/LTX-Video)）：`hf` 是 Lightricks 自己架在 ZeroGPU 的 demo，`comfy` 是本機 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)——demo 機器（AMD Radeon 8060S 內顯，32 GB）每段 79–121 秒，3 支 6 段約 9 分鐘。

每支只生成 2 段 5 秒而不是逐句生成 40 秒是刻意的：GPU 時間隨段數線性增加（雲端則是 $0.3–5/支），迴圈後觀感差異不大；數據本身用重繪的圖表比 AI 畫面更可信，所以圖表是疊在畫面上、不是被畫面取代。

**`hf` 什麼都不用裝**：`FINVID_AI_VIDEO=hf finvid run` 即可。下面是 `comfy` 的安裝：

```bash
# 一次性安裝（NVIDIA 照 ComfyUI 官方 README；AMD Windows 照下面，來自 AMD 的 ROCm 部落格）
py -3.12 -m venv D:\tools\comfyui-venv
D:\tools\comfyui-venv\Scripts\pip install -f https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ "torch==2.9.1+rocm7.2.1" "torchvision==0.24.1+rocm7.2.1" "torchaudio==2.9.1+rocm7.2.1"
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git D:\tools\ComfyUI
D:\tools\comfyui-venv\Scripts\pip install -r D:\tools\ComfyUI\requirements.txt   # 若 pip 想換掉 torch，用 -c 釘住 ROCm 版
# 模型（約 11.5 GB）
#   https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled.safetensors  -> ComfyUI/models/checkpoints/
#   https://huggingface.co/Comfy-Org/mochi_preview_repackaged/resolve/main/split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors -> ComfyUI/models/text_encoders/

# 啟動 ComfyUI（另一個終端）
D:\tools\comfyui-venv\Scripts\python D:\tools\ComfyUI\main.py --listen 127.0.0.1 --port 8188

# 跑
FINVID_AI_VIDEO=comfy finvid run          # PowerShell: $env:FINVID_AI_VIDEO="comfy"; finvid run
```

流程在 `pipeline/render/aivideo/`：`hf_space.py` 用 `gradio_client` 呼叫 Space 的 `/text_to_video`；`pixazo.py` 送 LTX 參數（幀數 8k+1、576×1024、8 步）到 `gateway.pixazo.ai`，輪詢 `polling_url` 到 `COMPLETED` 再抓 `media_url`；`minimax.py` 送 `video_generation`、輪詢 `query/video_generation` 到 `Success`、`files/retrieve` 拿下載網址（Hailuo 文字生影片沒有直式參數，只能出 16:9，`compose.py` 放大置中裁成 9:16，兩側會被裁掉）；`comfy.py` 的 workflow 是 ComfyUI API 格式的 JSON 模板（`workflows/ltxv_t2v.json`，可換成任何自己的 workflow，`FINVID_COMFY_WORKFLOW=` 指向即可），程式填入 prompt / 尺寸 / 幀數 / seed 後 `POST /prompt`，輪詢 `/history`，抓回檔案轉 h264。ComfyUI 沒開時會明確報錯，不會默默退回靜態卡。帳本記 `comfyui / gpu_second` 單價 $0，dry-run 也會估 GPU 秒數，所以「$0 但每支 90 秒」和「$0.27 但 20 秒」（MiniMax）可以放在同一張表比。

### 第 4 步的真實素材 B-roll（Pexels，免費）

AI 鏡頭是「生成的畫面」；`FINVID_BROLL=pexels` 則是把每一句台詞底下換成**真實的直式實拍素材**，來自 [Pexels Video API](https://www.pexels.com/api/)：

- **免費 key**：到 https://www.pexels.com/api/ 註冊即可拿到，填進 `.env` 的 `PEXELS_API_KEY`。沒填會在任何 TTS 之前擋下並指向這個網址。
- **額度**：每小時 200 次、每月 20,000 次搜尋（下載不計）。本片 3 支 clip 約 20 句，一次完整執行約 20 次搜尋；撞到 429 會明確報錯並提示額度，不會默默退回。
- **授權**：[Pexels License](https://www.pexels.com/license/) 可商用、免署名、可修改。
- **兩層快取**（都在 `data/_broll/`，跨影片共用、gitignore）：搜尋結果按查詢字串 hash 存成 `search_<sha1>.json`，下載的影片按 Pexels id 存成 `<id>.mp4`，裁到 `FINVID_BROLL_MAX_CLIP_SECONDS`（預設 8 秒）的版本是 `<id>_8s.mp4`。同一份腳本再跑一次是 **0 次 API 呼叫**；帳本記 `pexels / request` 單價 $0、數量是實際打出去的搜尋次數，dry-run 會印「~N Pexels requests, $0」。

**畫面怎麼排**：查詢字串就是第 3 步 Pass B 已經替每句台詞寫好的 `visual`（2–4 個英文素材關鍵字），不多叫一次 LLM。每句正文各配一段素材；開場 hook 若同時設了 `FINVID_AI_VIDEO`，就用**唯一一段** AI 鏡頭（`FINVID_AI_SHOTS_PER_CLIP` 被忽略，log 會說明），否則借第一個非空的 `visual`（或 `ai_shot` 描述）去搜。素材依播放順序交給同一個 `compose_clip`，所以場景窗剛好一句一段；某句搜不到就沿用上一場景的素材，畫面不會有空洞。圖表卡、標題／出處字卡、字幕的疊法跟 AI 鏡頭完全一樣。

**跟版權規則的關係**：素材是 Pexels 的授權影片，**不是原節目的任何畫面**；改寫後的文字、自製圖表、合成語音、「資料來源：TVBS《健康2.0》」字卡都不變，只是背景從靜態卡或 AI 畫面換成實拍。

```bash
FINVID_BROLL=pexels finvid run                       # PowerShell: $env:FINVID_BROLL="pexels"; finvid run
FINVID_BROLL=pexels FINVID_AI_VIDEO=hf finvid run    # hook 用 AI 鏡頭，其餘全部實拍素材
```

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
- 生成影片的「貴的那一步」（AI 影片生成）控制在固定用量：每支 clip 固定段數的 5 秒鏡頭（demo 是 1 段開場 + 每句真實素材），而不是逐句生成；而且只給通過篩選 + 反抄襲閘的 clip；免費 provider 是 $0 + 時間/額度，付費 provider 走同一個閘與 `FINVID_MAX_BUDGET_USD` 預算護欄——備援鏈會用鏈中**最貴**的那個 provider 預檢，所以 `hf,minimax` 在免費額度還沒用完時就已經把 MiniMax 的價格算進去。其餘畫面用 TTS + matplotlib + ffmpeg。帳本裡另記「若整支都用 Runway/Kling 類 API 會花多少」當對照（3 支約 $27.5）。

**「是否避免同一支影片重複處理」**
- `data/<video_id>/manifest.json` 記每個 stage 的設定 hash 與輸出檔。設定沒變、檔案還在就跳過。
- 改了第 2 步的設定，第 3、4 步自動失效重做，第 1 步不動。
- 第 3 步內部再細分：Pass A 的回覆另外以「模型 + 提示詞」hash 快取在 `03a_pass_a.json`，Pass B 觸發預算閘或改提示詞重跑時只付 Pass B。
- 同一支影片同時只允許一個執行：`data/<video_id>/.running.lock`（跨程序，CLI 與 web UI 共用；兩個終端同時 `finvid run` 第二個會被擋下，exit 4），逾 3 小時的殘留鎖視為當機遺留自動接手。

**額外護欄**
- `--dry-run`：全流程只估算，不呼叫付費 API、不下載音訊、不寫任何檔案（只向 YouTube 抓 metadata）。
- `FINVID_MAX_BUDGET_USD`：每次 API 呼叫前先算預估費用，超過就在呼叫前中止。
- 每筆 API 用量（秒數、tokens、字元）和單價都寫進 manifest，`finvid costs` 或 web UI 可查。

---

## 3. 版權與合法改寫

- **改寫非照抄**：Pass B 的提示詞明確要求重組句構、換用詞、數字不變。之後用程式檢查：腳本的 6 字元 n-gram 有多少比例出現在原逐字稿（門檻 15%）、最長共同子字串（門檻 12 字）。超標退回重寫一次，仍超標就丟到 `rejected_clips`，不進生成。
- **圖表自製**：腳本只帶數據（`chart.series[].points`），圖表由 matplotlib 從數據畫，不碰原影片任何畫面。
- **數字溯源閘**（$0，`pipeline/numbers.py`）：提示詞要求「數字只能來自逐字稿」是承諾，不是檢查。程式把逐字稿裡所有數字（含中文數字：一萬五、五千八百億、四成、百分之四十）解析成數值，每個圖表點的值必須能在逐字稿找到（Pass A 抽的 `data_points` 出自同一個模型，不算證據），找不到就丟掉該點（剩不到 2 點就整張圖不畫）；口白裡追不到出處的數字記在 `numbers_unverified` 供人工檢視（四則運算後的數字如「差了四千」是合理改寫，不判死）。demo 三支腳本的圖表點全部可溯源。
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
| AI 鏡頭 | HF ZeroGPU Space（免費雲端）＋ 本機 ComfyUI，同一個 LTX-Video 2B distilled 模型，可串成備援鏈 | $0、開源權重、不用信用卡；雲端 25 秒/段但有每日額度，本機 90 秒/段但無上限；ComfyUI 的 HTTP API 讓 workflow 可以整段換掉 | 付費雲端：MiniMax Hailuo（$0.08–0.27/段）、Kling（$0.18–0.42/段）、Veo（無免費 API）；其他免費：Pixazo LTX（preview 期） |
| 實拍素材 | Pexels Video API | 免費 key、可商用免署名、有直式篩選；查詢字串直接用 Pass B 已寫好的 `visual`，不多叫 LLM；搜尋與下載都快取，重跑 0 次呼叫 | Pixabay（額度較大但直式素材較少）、Coverr（無正式 API）、付費 Storyblocks |
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
  numbers.py        數字溯源：中文/阿拉伯數字解析，圖表點必須在逐字稿出現過
  stages/           s1_download s2_transcribe s3_script s4_render
  render/           tts chart compose fonts
  render/aivideo/   AI 鏡頭 provider：base.py（快取/帳本）、hf_space.py、pixazo.py、comfy.py + workflows/*.json、
                    minimax.py、kling.py（付費）、__init__.py 的備援鏈（預算閘用鏈中最貴的估價預檢）
  render/broll/     Pexels 真實素材（搜尋 + 下載雙快取）
  render/broll/     實拍素材 B-roll：pexels.py（搜尋/下載/裁剪快取在 data/_broll/、帳本）
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
- `data/demo/04_render.json` 的 `broll_paths` 指向 `../_broll/`（Pexels 素材的共用快取，gitignored，約 200 MB），fresh clone 裡不存在；成品 mp4 已把素材合成進去，這些路徑只是紀錄。
- 帳本只記每個 stage 最後一次成功執行；stage 中途失敗（例如第 2 支 clip 的 AI 鏡頭額度用完）時，前面已花的 TTS／已生成鏡頭的秒數不會進帳本，下次成功時鏡頭顯示為 cache hit 0 秒。`finvid costs` 另印 `runs[]` 的歷次總和作為對照。
- ffmpeg 失敗時只會看到 `CalledProcessError`，stderr 沒有轉出來；本機裝好 ffmpeg 後實務上沒遇過。
