"""Write the final Chinese report from retained measurements, including failures."""

import json
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
trials = {r["run"]: r for r in json.loads((R / "summary.json").read_text())}
parts = []


def add(s):
    parts.append(s.strip() + "\n")


def table(headers, rows):
    add(
        "|"
        + "|".join(headers)
        + "|\n|"
        + "|".join(["---"] * len(headers))
        + "|\n"
        + "\n".join("|" + "|".join(str(x) for x in row) + "|" for row in rows)
    )


def num(x, fmt=".3f"):
    return format(x, fmt) if x is not None else "—"


add("""
# FLASH 第三章首轮开发实验记录

> 这是审计修复前的历史数据。v1当前结论见 [v1报告](v1_report.md)，实现核对见 [忠实性审计](fidelity_audit.md)。旧条件及失败保留，不与修复后样本混合统计。

本轮完成了FLASH核心机制实现和两节点16卡的少量开发验证。真实15M参数MoE训练、监控正确性、持续退化自动定位、真实OOM/通信timeout、四种混合并行配置均已运行。**未复现论文全部性能和准确率指标**：正常强制剖析仍产生误候选，物理降频定位未成功，停机规则产生级联告警，早期条件下开销偏高；最终Gloo控制对照接近测量噪声。以下全部为本次实际记录，不填入论文数字作为实测结果。

本次没有读取以前的复现尝试。以中文论文第三章为规格，参照fresh clone的公开GreyHound源码；不实现SHIFT或后续优化。用户要求快速验证，因此每种故障通常一次，训练100–363步左右；不能由单次开发实验推出总体准确率或长时稳定性保证。

## 1. 实验对象与忠实度

两台H800共16卡，4条mlx5路径/节点；NCCL设置NET=IB并从日志确认实际采用IB/RoCE传输。主栈为PyTorch2.10/CUDA12.6/NCCL2.27.5、官方Megatron-core0.12.1、Rbeast0.1.25。论文使用H200、2.8a0/12.9/2.26.5；本次依用户要求缩小模型，因此不能声称重现13.3B Qwen训练的绝对吞吐。

主模型是真实15,083,776参数GPT MoE：4层、hidden256、8专家top2、1共享专家、SwiGLU、RMSNorm、GQA4、RoPE、原生All-to-All dispatch。真实Shakespeare字节next-token数据，从随机权重开始，FP32、原生Torch attention/norm、AdamW；micro batch2、sequence128、2个microbatches。完整来源与版本见[provenance.md](provenance.md)。

论文没有提供原代码，也未定义全部实现细节，故不能证明二进制等同原FLASH。已按详细设计实现并公开所有解释选择，见[spec.md](spec.md)：包括Rbeast/BOCD名称差异、20iter持久性与1–2iter延迟矛盾、CV分支、吞吐损失量纲、expertTP及分阶段标签。这些尚不能被一概认定为论文错误。

## 2. 设计、实现与验收
""")
table(
    ["论文要求", "实现", "验证与边界"],
    [
        [
            "NCCL透明拦截",
            "C++ LD_PRELOAD，InitRank/Config、Group、集合与Send/Recv",
            "真实训练与16分片checkpoint对照；未覆盖CommSplit/非NCCL通信",
        ],
        [
            "共享内存数据通道",
            "每进程POSIX环形缓冲，原子序号、写端锁、读端无锁",
            "正式profile统计两节点齐全，报告lost/errors",
        ],
        [
            "轻监控/按需剖析",
            "monitor仅记录元数据；profile复用CUDA Event，按group/stream计时",
            "monitor timed_events=0；profile真实计时；不支持capture时明确计数",
        ],
        [
            "简化communicator",
            "指针、unique-id哈希、global/local rank及size",
            "不依赖NCCL内部结构；按成员集合对应语义标签",
        ],
        [
            "Rbeast检测",
            "30点窗，概率.8，损失>.2，recovery<.75，右保护2，去重2",
            "持续/恢复/轻退化单测与训练实测；没有为结果降低阈值",
        ],
        [
            "动态profile",
            "max(2.2×最新迭代秒数,20s)，本地0.1s轮询",
            "本次短迭代均为20秒采样，故总定位不能说仅2–3个迭代",
        ],
        [
            "两阶段定位",
            "标签内CV.1选max_avg_latency最大组，组内取均值最小rank，低于均值20%",
            "跨标签累计；EP数据记录但不参与定位；不读取注入目标",
        ],
        [
            "fail-stop",
            "2×最新迭代时间无NCCL入口",
            "OOM/timeout有真实日志；出现额外rank告警，首轮未记录告警时进程存活状态",
        ],
        [
            "框架接入",
            "共同的迭代/并行状态两个回调，无Megatron源码改动",
            "PyTorch、Megatron主实验；DeepSpeed单独smoke；不宣称零适配",
        ],
    ],
)
add("""
四项单元测试全部通过，包含持续/恢复/轻退化、两阶段反向rank特征、低CV/缺成员拒绝，以及SP RMSNorm的分片/非分片输出、输入梯度和参数梯度等价性。原生norm适配仅解决当前框架无TE/Apex时SP初始化限制，没有禁用SP。

16rank的BF16 GEMM/AllReduce预检通过。正式baseline/monitor两次检查的16个模型分片均finite且最大参数差为0；final对照为160步，证据见 `results/paper_monitor_final/checkpoint_check_node*.json`。checkpoint本体保留服务器，不重复下载。

## 3. 监控与剖析开销

端到端吞吐排除前5步，使用剩余连续墙钟时间（含上报和fixture辅助通信），不以单步倒数平均代替整体吞吐。三组结果全部列出。早期两组曾存在实验驱动未停止observer的问题；修复和清理后额外进行final对照，未用新结果覆盖旧结果。
""")
perf = []
for b, m, label in [
    ("paper_baseline16", "paper_monitor16", "第一组120步"),
    ("paper_baseline16b", "paper_monitor16b", "第二组120步"),
    ("paper_baseline_final", "paper_monitor_final", "清理后final160步"),
]:
    x, y = trials[b]["steady_tokens_per_second"], trials[m]["steady_tokens_per_second"]
    perf.append(
        [label, num(x, ".2f"), num(y, ".2f"), num(100 * (1 - y / x), ".3f") + "%"]
    )
table(["运行", "baseline tokens/s", "monitor tokens/s", "吞吐损失"], perf)
if "paper_gloo_monitor" in trials:
    b, m = trials["paper_gloo_baseline"], trials["paper_gloo_monitor"]
    add(
        f'**最终方法修正对照（Gloo退出控制）**：去掉fixture每步辅助NCCL AllReduce，改用独立Gloo CPU组协调结束。baseline={num(b["steady_tokens_per_second"],".2f")}、monitor={num(m["steady_tokens_per_second"],".2f")} tokens/s，吞吐损失{num(100*(1-m["steady_tokens_per_second"]/b["steady_tokens_per_second"]),".3f")}%。以上三组旧结果含辅助NCCL，不删除或与最终条件混合平均。最终代码默认Gloo，可用--stop-backend nccl恢复旧fixture。该修正去除测试干扰，没有改变FLASH算法。'
    )
if "paper_gloo_healthy" in trials:
    r = trials["paper_gloo_healthy"]
    add(
        f'移除辅助NCCL后的正常强制剖析：自动profile {r["automatic_profiles"]}次、强制{r["forced_profiles"]}次、候选{r["predicted_ranks"]}、fail-stop告警{r["failstop_ranks"]}。本轮正常定位结论优先参考此项；旧rank0误候选受辅助通信影响，不能直接归因论文；本次仍有其他误候选。最终16分片参数检查同样最大差0，见paper_gloo_monitor/checkpoint_check_node*.json。'
    )
    samples = [
        json.loads(line)
        for line in (R / "paper_gloo_healthy" / "train_rank0.jsonl")
        .read_text()
        .splitlines()
    ]
    samples = [x for x in samples if x["kind"] == "iteration"]
    pre = statistics.median(x["time_ms"] for x in samples if 20 <= x["iteration"] < 35)
    prof = statistics.median(
        x["time_ms"] for x in samples if 50 <= x["iteration"] < 100
    )
    add(
        f"最终正常样本同一口径的采样前/采样期间迭代中位数为{pre:.3f}/{prof:.3f}ms，耗时变化{100*(prof/pre-1):+.2f}%，未测得明确正开销，不解释为剖析加速。Gloo控制本身增加fixture墙钟成本，因此绝对吞吐不可跨旧/新fixture直接比较，monitor与baseline始终使用相同退出协议。"
    )
if "paper_gloo_delay45" in trials:
    r = trials["paper_gloo_delay45"]
    add(
        f'同条件持续退化复测：实测损失{num(100*r["actual_throughput_loss"],".1f")}%，自动profile {r["automatic_profiles"]}次，候选{r["predicted_ranks"]}，触发{num(r.get("detection_seconds"))}s、诊断{num(r.get("localization_seconds"))}s，fail-stop告警{r["failstop_ranks"]}，lost={r["lost"]}、probe errors={r["probe_errors"]}。'
    )
add(
    "AllReduce为ABBA顺序、4–128MiB，各20次预热、3批×100次，少于论文500预热/10,000次。下表统一取rank0的批次平均，所有16rank原始数据仍保留。负开销反映噪声，不能解释为监控使通信加速。"
)
bench = []
for i, size in enumerate([4, 8, 16, 32, 64, 128]):
    values = []
    for suffix in ["1", "2"]:
        b = json.loads(
            (R / f"paper_batch_base{suffix}" / "allreduce_rank0.json").read_text()
        )[i]["mean_ms"]
        m = json.loads(
            (R / f"paper_batch_mon{suffix}" / "allreduce_rank0.json").read_text()
        )[i]["mean_ms"]
        values.extend([num(b), num(m), num(100 * (m / b - 1), "+.2f") + "%"])
    bench.append([size] + values)
table(["MiB", "base1 ms", "mon1 ms", "开销1", "base2 ms", "mon2 ms", "开销2"], bench)
add("""
结果未支持论文“全部小于0.8%”的上界；短测小消息波动明显。128MiB的两对观察值低于1%，仍不能以有限样本确立严格上界。

正常强制profile复测 `paper_profile_healthy_fixed`：没有自动profile、没有fail-stop告警，但定位器给出无故障rank0候选（score约.912）。这属于强制运行定位器的误候选，不能混算自动门控误触。采样前iteration20–34中位数81.433ms，采样期间iteration50–99中位数87.672ms，延迟约+7.66%；该窗口比较是开发观察值，非重复配对统计，未达到论文剖析开销≤5%的描述。短正常运行没有自动误触不等于验证了论文数周零误触。

![Performance](../results/figures/performance.png)

## 4. 持续与瞬时退化

GPU delay使用真实GPU sleep，在rank8完成计算、同步梯度之前注入；目标仅用于测试驱动和离线评分。名义严重度先按稳定迭代校准，但部分等待可消耗原有同步空隙，因此以实际损失分类。实测损失=1−注入前15步迭代中位数/注入后第5–19步中位数，包含剖析影响；是短窗口观察值。
""")
fault_rows = []
for run in [
    "paper_delay5",
    "paper_delay25",
    "paper_delay_boundary",
    "paper_delay45",
    "paper_delay65",
    "paper_delay85",
    "paper_transient",
]:
    r = trials[run]
    fault_rows.append(
        [
            run.removeprefix("paper_"),
            (
                num(100 * r["actual_throughput_loss"], ".1f") + "%"
                if run != "paper_transient"
                else "单步500ms"
            ),
            r["automatic_profiles"],
            r["predicted_ranks"],
            (
                f'{r.get("tp",0)}/{r.get("fp",0)}/{r.get("fn",0)}'
                if run != "paper_transient"
                else "不适用（瞬时）"
            ),
            num(r.get("detection_seconds")),
            num(r.get("localization_seconds")),
            len(r["failstop_ranks"]),
        ]
    )
table(
    [
        "运行",
        "实际损失/扰动",
        "自动profile",
        "慢候选",
        "慢定位TP/FP/FN",
        "触发s",
        "诊断s",
        "停机告警数",
    ],
    fault_rows,
)
add("""
持续退化在实测约27.6%、40.8%、62.9%、84.7%四次中均自动定位rank8且无其他慢候选；这只是4次观察结果，不推广为总体100%精确率。名义25%实际约16%，没有达到严格20%门槛，不能直接记作“25%全局退化漏检”。实际约3%的轻扰动也未触发，不能声称已复现论文5%召回。

单步500ms没有自动进入慢故障剖析，符合瞬时恢复过滤，但严格无调用规则对16rank均产生fail-stop告警。85%持续迟延同样有16停机告警：这是两条检测路径的类型混淆，不能只展示正确的慢定位而隐藏停机误报。

上述触发/诊断时延从fixture注入事件到控制事件计算，包含0.1s轮询和Rbeast运行，不是单次NCCL时延。跨节点wall clock未做PTP级校准，表中毫秒位仅为记录分辨率，时延按近似观察解释。20秒profile下限主导了总诊断时间。

## 5. 真实停机
""")
stoprows = []
for run in ["paper_oom_clean", "paper_timeout_clean"]:
    r = trials[run]
    alerts = r["failstop_ranks"]
    stoprows.append(
        [
            run.removeprefix("paper_"),
            r["exits"],
            num(r.get("failstop_detection_seconds")),
            int(8 in alerts),
            len([x for x in alerts if x != 8]),
            int(8 not in alerts),
        ]
    )
table(
    ["故障", "两节点退出码", "目标告警延迟s", "目标告警", "其他rank告警", "目标漏告警"],
    stoprows,
)
add("""
OOM真实申请158.36GiB显存，GPU79.18GiB，保留CUDA OutOfMemoryError。timeout由独立测试注入库在一次ncclAllReduce中阻塞110秒，训练框架90秒watchdog实际报 `Timeout(ms)=90000`。Probe/控制器不读取该注入开关。两项训练非零退出符合注入预期。证据收齐后曾终止清理过慢的残留worker，动作独立记录，不作为检测时刻，也没有实现恢复。

若把全部告警都当作“唯一注入根因预测”，每次只有1个目标命中、15个非根因告警；但这不能等同于15个健康进程误报。首轮未记录告警时的进程存活状态，torchrun/NCCL可能已级联终止同伴，实际停机进程检测的精确率无法据此计算。原规则检测通信沉默，不能区分同步等待与实际退出，也未实现唯一根因识别。

## 6. 物理锁频及四种混合并行

每次请求在节点2的GPU0（global rank8）锁频，并在finally恢复。下表保留请求值和一次查询值；1965MHz请求实际采样1830MHz，1600请求采样1605MHz，不能将请求值当精确运行频率。所有其他GPU保持原环境动态频率。
""")
eq = list(csv.DictReader((R / "paper_equations.csv").open()))
clockrows = []
for f in [1965, 1600, 1200, 900, 600, 345]:
    run = "paper_clock600_ep8_tp1_pp1" if f == 600 else f"paper_clock{f}"
    r = trials[run]
    hw = json.loads((R / run / "clock_injection.json").read_text())
    queried = next(
        (
            e["output"].strip().splitlines()[-1].split(",")[0]
            for e in hw
            if "current.memory" in e["command"]
        ),
        "—",
    )
    scores = [float(e["eq32"]) for e in eq if e["run"] == run and e["rank"] == "8"]
    clockrows.append(
        [
            f,
            queried,
            num(100 * r["actual_throughput_loss"], ".1f") + "%",
            r["automatic_profiles"],
            r["predicted_ranks"],
            num(statistics.median(scores) if scores else None),
        ]
    )
table(
    [
        "请求MHz",
        "实测查询",
        "实际吞吐损失",
        "自动profile",
        "forced候选",
        "目标式3-2中位数",
    ],
    clockrows,
)
add("""
这些小模型锁频试验未产生达到20%门槛的全局损失，因此无自动profile，不能算自动检测成功。强制采样也未成功定位rank8，产生了其他候选；未复现论文随频率下降而得分显著单调增加的现象。小模型计算占比、同步空隙、不同算子栈可能影响信号，但目前没有证据将全部差距归于硬件或论文。

旧fixture在iteration60允许forced采样，锁频完成事件的记录在部分运行中比profile开始晚约0.8秒，20秒窗口包含少量转换区间。因此这些是探索性的强制采样数据，不能声称严格稳定锁频后的配对测量。已修复以后重跑入口：锁频命令完成后才放行forced。没有篡改既有结果或将forced提前量写成负的自动检测时延。

式(3-1)使用每组rank延迟的算术平均，与同标签各组平均比较；式(3-2)对所有rank计算。导出 `paper_equations.csv`，包括未通过阈值的项。控制器第一阶段选组使用max_avg_latency，`clique_scores.csv`是该实际筛选分数，两者不能混称。
""")
toporows = []
for tp, pp, ep, suffix in [
    (1, 1, 8, ""),
    (2, 1, 4, "_fixed"),
    (1, 2, 4, ""),
    (2, 2, 4, "_fixed"),
]:
    run = f"paper_clock600_ep{ep}_tp{tp}_pp{pp}{suffix}"
    r = trials[run]
    score = sum(
        c["score"]
        for d in r["diagnoses"]
        for c in d["result"]["suspects"]
        if c["rank"] == 8
    )
    toporows.append(
        [
            f"EP{ep}/TP{tp}/PP{pp}/SP{tp}",
            r["exits"],
            r["predicted_ranks"],
            num(score),
            num(100 * r["actual_throughput_loss"], ".1f") + "%",
        ]
    )
table(["拓扑", "两端退出", "forced慢候选", "目标入选累积分数", "实际损失"], toporows)
add("""
四组真实训练都成功；目标入选分数0表示未通过定位器的筛选，不表示其原始式(3-2)值全为0。没有复现论文图3-10的1.01–1.15累积分数。expertTP设1，PP阶段DP/TP标签分开，EP不参与定位，故本次也不能保证论文所述2/4/4/5组证据数量。初次TP2因原生norm不支持SP在初始化失败，修复后重新使用新编号运行；初始化失败不是FLASH漏检。

![Faults](../results/figures/faults.png)

## 7. 背景通信、框架兼容与官方基线
""")
if "paper_gloo_clock600" in trials:
    r = trials["paper_gloo_clock600"]
    add(
        f'最终600MHz控制复测移除辅助NCCL，且锁频完成后才放行forced：两端退出{r["exits"]}、实测损失{num(100*r["actual_throughput_loss"],".1f")}%、自动profile {r["automatic_profiles"]}次、强制{r["forced_profiles"]}次、候选{r["predicted_ranks"]}、停机告警{r["failstop_ranks"]}。旧六档扫描及四拓扑保留作探索证据；未对全部条件按新fixture重复，不能证明旧差距均源自原算法。图中箱体来自同一运行的不同通信组，并非多次独立试验。'
    )
if "paper_background_devicefixed" in trials:
    r = trials["paper_background_devicefixed"]
    add(
        f'背景通信 `paper_background_devicefixed`：两端退出{r["exits"]}，实测短窗吞吐损失{num(100*r["actual_throughput_loss"],".1f") if r["actual_throughput_loss"] is not None else "—"}%，自动profile {r["automatic_profiles"]}次，慢候选{r["predicted_ranks"]}，停机告警{r["failstop_ranks"]}。独立双rank跨节点128MiB AllReduce，30秒，分别共享训练节点GPU0；同时有GPU/内存带宽争用，不能定性为纯网络链路故障。两个背景rank以一致collective计数退出，JSON中记录调用数、持续时间和数值正确性。'
    )
if (R / "paper_deepspeed_compat" / "execution.json").exists():
    e = json.loads((R / "paper_deepspeed_compat" / "execution.json").read_text())
    add(
        f'DeepSpeed0.19.7兼容性运行：两端退出{e["exits"]}。仅stage0 dense next-byte短测，真实forward/backward/AdamW与NCCL；只能验证通用probe和共同回调接入，未覆盖DeepSpeed MoE/ZeRO。完整16rank日志保留，不能将其当主MoE验证。'
    )
if (R / "greyhound_official16_gated" / "execution.json").exists():
    e = json.loads((R / "greyhound_official16_gated" / "execution.json").read_text())
    add(
        f'官方GreyHound基线：fresh公开commit已编译，MAX_DEVS/拓扑总rank/路径适配补丁完整保存。前两次启动暴露并行状态发布竞争及训练结束早于precheck，均保留LIMITATION。最终 `greyhound_official16_gated` 预填准确的上游组键，先预热通信组、服务503-byte原验证协议并等待真正precheck_done，随后注入训练GPU delay。两端退出{e["exits"]}；原版完成慢事件上报、profile、主动计算/通信验证和缓解规划入口，local报告rank8–15，global判为comm，日志另有batch size缺失提示。详见该目录的baseline_observation.json和控制器日志。训练sleep在暂停训练后的独立计算验证中不持续，加上旧PyTorch/NCCL/Megatron与不同模型，不能由此建立与现代FLASH同条件准确率/性能对比，也不填充论文对照数值。上游缓解行为属于该官方基线自身，不是本轮添加FLASH恢复。'
    )
add("""
## 8. 实现问题、证据和未完成的论文级验证

开发过程发现并修复的复现错误：observer子进程PYTHONPATH漏依赖导致停止命令失败；控制器在耗时聚合前读取心跳造成陈旧采样；背景通信两端按独立墙钟退出造成collective次数不一致；TP2/SP缺原生norm适配；重复使用OOM编号。相应旧运行标INVALID/LIMITATION，保留排障证据，用新编号复测。不能将这些错误归咎于论文。

剩余限制：没有做数周误触验证、论文10,000次微基准、多次频率箱线统计、13.3B/Muon/TE同栈训练、纯链路拥塞隔离和同栈GreyHound完整精确率/召回率对照；属于按用户“小模型、快速、少次数”范围缩减或当前未达成的对照条件，不声称完成原论文全部实验统计。

早期fixture每步额外一次world MIN AllReduce用于一致结束，其成员集合可能与dense DP重合；标签匹配不会区分消息语义/大小。这是实际方法干扰，最终代码改用Gloo结束协调，并补测吞吐、正常剖析、持续退化和600MHz锁频。未重复全部旧条件，所以旧频率/拓扑定位失败尚不能排除该干扰。已保留hash和原始记录。所有profile均使用原始持续时间而非带宽归一化，没有加入额外诊断算法来追求分数。当前结果不能排除论文未定义细节与fixture选择带来的影响。

项目入口见[runbook.md](runbook.md)，设计追踪见[spec.md](spec.md)。`results/summary.json`、`trials.csv`、`clique_scores.csv`、`rank_scores.csv`、`paper_equations.csv`均可从原始JSON重建。每个有效新运行保存配置、源码/库哈希；单元测试、checkpoint、最终服务器空闲/复位证据另存。服务器保存全部模型检查点。下一阶段是否改进由复现结果讨论决定，本轮没有继续调优检测算法。
""")
(ROOT / "docs" / "report.md").write_text("\n".join(parts))
print(ROOT / "docs" / "report.md")
