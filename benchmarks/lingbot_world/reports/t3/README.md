# T3 interaction latency profile and optimization

Issue: <https://github.com/vllm-project/vllm-omni/issues/9565>.
This report was prepared with AI assistance from local measurements on 2026-11-08.

## 总结

已找到并实现一个有实测收益的优化：为已压缩的 MP4 关闭 WebSocket
permessage-deflate，减少发送期间持有 output credit 的等待。
编码异步化另有事件循环响应性收益。当前证据不足以宣布 T3 全部验收通过。

| Issue 方向 | 已完成 | 结果与缺口 |
|---|---|---|
| T3_TRACE：复现延迟和等待 | 双卡 H200 GPU trace、客户端时间线、编码/发送/credit 等待记录，以及真实浏览器 tagged-frame 测量 | 定位重复压缩造成的交接等待；未严格复现 frozen 配置 |
| T3_OPT：减少可避免等待 | 每条流使用单个编码工作线程；Python 示例和 benchmark 关闭重复压缩 | 编码异步化改善响应性；关闭压缩使模拟控制响应均值改善约 5.5% |
| T3_VALIDATE：响应与播放权衡 | 每组 3×40 chunks、收紧缓冲实验、浏览器短测、解码和传输一致性检查 | 基础场景有证据；velocity 短按/长按、弱网断连和长期运行尚不充分 |

## 实验环境与端点

- 源码基线：`aa04b1caf72fed67e1fc692b5b7f8b39cb51f13f`，加本分支工作区改动。
- 模型：`lingbot-world-v2-14b-causal-fast-diffusers`；缓存 revision：
  `59cccf49f2d2dd27418ae7a04b82b10868d455c2`。
- 两张 H200 NVL，每张 143771 MiB；实际互联为跨 NUMA 的 SYS/PCIe，非 NVLink。
  同机已有 Qwen-Image 服务，每卡约占用 43 GB，实验期间保留。
- vLLM 0.32.0、PyTorch 2.13.0+cu130、Transformers 5.19.0；独立环境
  `/workspace/t3-venv`。FlashInfer cubin 版本存在不匹配警告，ring kernels 不可用；
  本实验使用纯 Ulysses SP2、ring degree 1、FA3。
- [配置](deploy.yaml)：832×480、4 denoising steps、12 FPS、seed 42、选择性 FP8、
  last-step KV reuse、fused VAE。输入为本地保存的 nature boardwalk 图片。
- 正式 Python 对比每组 3×40 chunks，camera target 控制在模拟 playhead
  6/12/18/24/30 秒发送；反馈间隔 50 ms。稳态 cadence 排除最初六块及末块。
- Python 端点是首个带控制标签 chunk 首帧的**模拟播放时间**，不是浏览器显示时间。
  浏览器另行使用 `requestVideoFrameCallback` 的 `presentationTime` 减控制发送时间，
  均来自同一浏览器 performance 时钟，不混减服务端时钟。

这些是匹配的本机前后对比，不是 frozen H200 基线的精确复现，也不是 950 qualification。

## 各调优方向结果

| 尝试 | Before → After | 判断 |
|---|---|---|
| 编码移到串行工作线程 | 事件循环 lag p99：98.91 → 56.94 ms；控制到模拟呈现：2518.59 → 2530.40 ms | 响应性改善，端到端无收益 |
| 关闭 WebSocket 重复压缩 | chunk 间隔：1380.38 → 1310.74 ms；控制到模拟呈现：2518.10 → 2378.51 ms | 当前最明确的有效优化 |
| 关闭重复压缩的播放影响 | 每 40 chunks 卡顿总时长：15.22 → 12.49 s | 减少约 17.9%，仍慢于实时生成 |
| lead cap 从 3 降到 1.5 s | 24-chunk 探索实验控制响应约 2575 → 1683 ms；卡顿总时长 9.54 → 21.94 s | 代价明显，未采用为默认值 |
| 提前释放 output credit / 重叠交接 | 未修改 credit 所有权 | 需要额外证明缓冲有界、顺序及取消清理正确 |
| GPU kernel 调优 | 已保存 attention、NCCL、FP8 GEMM 时间分布 | 本轮无 kernel 优化，无 NCU counters |

编码实验每组先 warmup 12 chunks。压缩实验在同一服务进程中先 warmup 16 chunks、
运行 16-chunk 诊断，再依次执行 no-deflate 3×40 和 deflate 3×40；两组均启用异步编码。
实验顺序未随机化，不能把压缩收益归于编码异步化。

压缩实验 WS send 均值约 72.59 → 0.64 ms，观测到的 output-credit blocked
时间约 4.51 → 1.53 s/session。正式实验观测到的 blocked mask 只包含 outstanding
output，其他门限未出现阻塞。该时间是调度器检查门限时的观测状态区间，不能当成精确的
资源生命周期，也不能与客户端延迟相加。重新压缩 MP4 仅节省约 0.65% 字节。

短 GPU capture 的 rank 0 kernel window 约 2509 ms；attention 累计 766.7 ms、
NCCL 683.7 ms、FP8 GEMM 375.6 ms。不同 stream 的 kernel 可重叠，不能作为互斥占比。
GPU profiler 已保存原始 traces，但 stop 后的算子表导出挂起；表格由原始 JSON 解析。

数据：[编码对比](evidence/encoder-comparison.json)、
[压缩对比分布](evidence/transport-comparison.json)。

## 真实浏览器结果

每组一次 24-chunk 探索实验、三个 camera target 控制，配置一致但服务重启并 warmup。
使用 headless Chromium，禁用 GPU rendering；端点是提交给 compositor，非物理屏幕显示，
也不证明视觉运动语义正确。浏览器实验通过本地 hook 禁用服务端压缩；本分支只修改
Python 客户端设置，未修改生产服务端或浏览器默认设置。没有记录浏览器握手扩展。

| 指标 | 默认压缩 | 禁用压缩 |
|---|---:|---:|
| 三个控制响应，ms | 2750.4 / 2500.3 / 2516.6 | 2618.0 / 2351.1 / 2334.1 |
| 平均控制响应，ms | 2589.1 | 2434.4 |
| 播放 waiting 总时长，s（不含初始启动） | 9.861 | 8.218 |
| waiting 次数 | 23 | 23 |
| 浏览器报告 dropped frames | 23 | 23 |
| 帧回调数 | 285 | 285 |
| 控制映射到带标签首帧 | 3/3 | 3/3 |

两组所有 chunk 的首个回调 mediaTime 都等于 chunk 起始时间（容差 1 微秒），
没有拿后续帧代替首帧；回调 mediaTime 严格递增。回调计数不能替代浏览器丢帧统计。
[浏览器检查数据](evidence/browser-comparison.json)。

## 正确性、测试与限制

- handler 测试文件：42 passed；新增阻塞编码期间控制/stop 可处理、取消后串行 close、
  close 中取消和失败记录测试。原始同步实现无法通过响应性负对照。
- benchmark 测试：77 passed。
- 六段压缩对比 GPU 视频均解码出预期 477 帧，PTS 严格递增：
  [视频检查](evidence/video-validation.json)。
- 实际 CPU WebSocket 回放在压缩开/关、立即/延迟读取四种组合中，82 条消息顺序与内容完全一致，
  41,971,695 媒体字节 SHA-256 相同：[传输检查](evidence/transport-validation.json)。
  这是传输一致性，不是不同交互生成会话之间的像素一致性，也不等于真实弱网压力测试。
- 改动范围 Ruff、格式及 diff 检查通过。mypy 的既有 `OmniTextPrompt` 缺字段错误在原始代码
  中也可复现；不能声明 mypy 全绿。未运行完整仓库 CI 或 CUDA L2。
- 每条流的 encode/close 在同一 worker 上串行执行；取消后 close 排在正在运行的 encode 后，
  事件循环不等待线程退出。原生编码器永久挂起时无法强制终止线程，最终清理依赖原生调用返回。
- 尚需验证：多轮随机顺序浏览器测试、不同 chunk 相位的短按和长按 velocity 控制、实际带宽限制、
  断连恢复、长期资源行为、物理显示以及视觉控制语义。当前不能保证所有场景正确。

## 相关 PR 的启发

- [#8721](https://github.com/vllm-project/vllm-omni/pull/8721)：credit 覆盖下游消费，
  下一整块参与 lead admission；不能直接删除 credit 等待。
- [#8952](https://github.com/vllm-project/vllm-omni/pull/8952)：采用 tagged-frame 浏览器端点；
  收紧缓冲有卡顿代价，target 控制不能代替 velocity 短按/长按验证。
- [#8810](https://github.com/vllm-project/vllm-omni/pull/8810)：避免控制晚一个 chunk 的
  penultimate-step return 修复已存在，不算本轮新收益。
- [#9377](https://github.com/vllm-project/vllm-omni/pull/9377)：上述修复的 backport，非独立优化。

## 本地完整 profile 与复现材料

完整约 760 MB 的原始材料保留在工作区 `artifacts/t3_profile/`，未纳入 Git 提交。
本目录提交轻量结果和配置，JSON 中的原始文件路径指向该本地目录。

- `report.html`：离线总报告及 session 时间线。
- `REPORT.md`、`handoff/REPORT.md`、`browser/REPORT.md`：详细结论和运行说明。
- `gpu_baseline/reports/20261108-073038_stage_0_rep_0_diffusion_1794123038/trace_rank*.json.gz`：
  GPU traces；`gpu_baseline/analysis/`：kernel 汇总。
- `handoff/handoff_trace.json`：可导入 Perfetto/Chrome 的 host 等待时间线。
- 每个 `session_*`：原始协议事件、trace、MP4、summary。
- `reproduce.sh`、`profile_client.py`、`browser/run_browser.py`、`browser/player.html`：
  复现脚本；`provenance/`：版本、硬件和输入哈希。

拷贝整个本地 profile 目录可保留 HTML 相对链接。实验服务已关闭，原有 Qwen 服务保留。
