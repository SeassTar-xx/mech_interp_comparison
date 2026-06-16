# Day-1 汇报：NLA / VPD 可用性检查

今天主要完成了 NLA 和 VPD 两条线的 day-1 feasibility check，原则是只使用已有 checkpoint / artifact，不重新训练、不 fine-tune。

## 已完成进展

**NLA**：已经跑通 Qwen2.5-7B-Instruct layer 20 的 NLA AV/AR 流程。base model 使用 `/mnt/20t/xuhaoming/models/Qwen2.5-7B-Instruct`，AV/AR checkpoint 都已在本地模型目录。最终 SGLang 版本处理了 `16` 条 mini prompts，覆盖 safety / bias / sycophancy / evaluation awareness 等类别；输出包括 base model answer、top tokens、NLA explanation、AR reconstruction score。SGLang 问题也已修复，结束后 server 已关闭。

**VPD**：已经 clone Goodfire `param-decomp`，并切到 paper release tag `vpd-paper` 做 artifact loading。已确认 canonical run `goodfire/spd/runs/s-55ea3f9b` 可下载并加载，VPD checkpoint 大约 2.71 GiB；同时下载了所需 target pretrain artifact。现在可以在 CPU 上成功构造 `ComponentModel`，并完成了一个很小的 pronoun prediction smoke test：`The princess lost her crown.` 的 top next tokens 中出现 `She/Her`，`The prince lost his crown.` 中出现 `He/His`。没有进行任何 VPD 训练。

## 初步判断

NLA 更适合继续做 Qwen2.5 safety / bias / sycophancy 的同模型 prompt-level comparison，因为它直接解释 residual activation 并输出自然语言。VPD 更适合作为 parameter-level canonical case study，因为当前可用 artifact 对应 Goodfire 小模型，不适合直接和 Qwen instruction behavior 做 apples-to-apples benchmark。

## 当前 blocker / 风险

NLA 的主要风险是 explanation 很可读，但不一定 causal，后续需要 claim extraction、prediction consistency 和 SR baseline。VPD 的主要 blocker 是还没有抽取 top causal components / ablation / attribution graph；另外部分 cluster mapping / postprocess artifact 可能在 Goodfire 内部路径，不一定公开。

## 建议下一步

1. NLA 先固定 Qwen2.5-7B layer 20，把 mini prompts 扩展到 safety / bias / sycophancy 每类 50-200 条。
2. 用同一批 prompts 加 SR baseline，并保留 base output、top tokens、NLA explanation、AR reconstruction score。
3. VPD 优先复现一个已有 case study（pronoun prediction 或 bracket closing），不要从零训练。
4. 如果组内有 VPD artifact / cluster mapping / attribution graph 镜像，优先接入；如果没有，先把 VPD 放成 method-level case study。

## 明天想确认的问题

- 组内是否已有 VPD checkpoint / postprocess / graph artifact 镜像？
- NLA 是否先固定 Qwen2.5-7B layer 20？
- VPD 是否按 parameter-level case study 展示，而不是强行做 safety/bias/sycophancy 同模型 benchmark？
- 第一版 NLA 需要每类多少样例？
- 是否马上加入 SR baseline？
