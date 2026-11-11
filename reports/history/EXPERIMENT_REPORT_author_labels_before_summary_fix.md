# 跨来源 C–N 偶联可行性预测：实验报告

本文件是计算实验记录与证据核验，不是论文；未投稿、不承诺录用。

当前状态：**CONTINUE**。主候选固定为 `B_A_S`；接受规则结果 `INCREMENT_CRITERION_NOT_MET`；证据类别 `no_increment_criterion_met`。

这是阶段报告；仍有实验或核验待完成。

若仅secondary达标，明确表示主B_A_S确认未成功、预注册简化方案获得多重校正支持；这不改写主假设，不从外层结果追认新的独立确认。简化方案还必须在宏AUROC上不劣于B_bal_S。

## 数据、问题与固定协议

目标为未见来源且芳基、胺各自均未见时，判断记录的 Yield 是否至少为10；恰好10为正例。不是精确产率、高产率或未实测条件推荐的验证。
原始行数 `27497`；主数据行数 `23185`。合格主来源数 `4`；所有来源统计与逐行处置见 [DATA_AUDIT](../DATA_AUDIT.md)。

机器可核验冻结记录见 [FINAL_EVAL_LOCK](../FINAL_EVAL_LOCK.json)，SHA256 `b7005aff7fa71ab452015047b8b3a0d0aa0efa718fcdb937915d1267f6576a84`。EXPERIMENT_PROTOCOL.md保留冻结前的working标题；实际冻结状态、训练预算、候选方法、seed及内层选择规则由该只读锁和其绑定制品给出，外层评价按该锁执行。

| fold | source | eligible | test_rows | test_pairs | test_aryls | test_amines | inner_splits |
|---|---|---|---|---|---|---|---|
| az_eln_2023 | AZ ELN 2023 | True | 607 | 376 | 230 | 207 | 2 |
| dreher_2014 | Dreher 2014 | False | 0 | 0 | 0 | 0 | 2 |
| dreher_2016 | Dreher 2016 | False | 0 | 0 | 0 | 0 | 2 |
| hitea_2024 | HiTEA 2024 | False | 1387 | 15 | 14 | 13 | 2 |
| jnj_hte_2024 | JNJ HTE 2024 | True | 8016 | 546 | 90 | 65 | 2 |
| merck_2025 | Merck 2025 | True | 2668 | 2668 | 335 | 8 | 2 |
| roche_2023 | Roche 2023 | True | 1680 | 65 | 28 | 18 | 2 |

外层LOSO的整个开发集包含验证/校准，测试两类底物与其实体集合均无交集；同对条件和重复实验不随机跨边界。独立审计从冻结records与splits重新核验，而非复用训练检查函数。
原始存储指纹不可解析，已重算。R0为不使用产品的DRFP；R1_substitute为作者启发的底物Morgan表示＋公开RDKit21试剂描述符，是声明的强替代对照，未准确重建作者底物–产品DRFP＋原QM生产模型，不能声称超过原作模型。全部神经方法使用同样的RDKit21输入。

Tokenizer显式调整：`{"added_token_ids": [591, 592], "added_tokens": ["->", "<-"], "embedding_initialization": "copy_original_minus_embedding", "extended_vocab_sha256": "b7c0c14a53d991b68091e910b3e5b147a8e8290cf754f9afc703e64c0d13f0f7", "extended_vocab_size": 593, "original_vocab_sha256": "399868a653e85549ce150af4a1f6cd2776240c6a3d951ced2120565f00f027fd", "original_vocab_size": 591, "regex_sha256": "a013643a65bc228c8133ca2c0908e61a78453908735906f44b2722d6f7516b47", "rule_sha256": "aaf1880a7c161babfd9dc7eecb448b39485f0dac256d6eba6c49aba93d70663e", "version": "rxnfp_dative_v1"}`。两个配位箭头token扩充原词表，原ID保留，新embedding复制原减号embedding；所有神经对照一致。无静默截断或删除关键试剂。

## 主表、逐来源结果与消融

每个seed独立计算AUROC/AP/Brier，再对seed平均，最后对合格来源等权平均。AP使用非插值average precision；不同来源不混合计算主AUC。cal_brier是内部温度校准后的辅助概率误差，不能解释为排序改善。

| model | n_sources | auroc | auprc | brier | cal_brier | worst_source_auroc | low | high | invalid_fraction |
|---|---|---|---|---|---|---|---|---|---|
| R0_RF | 4 | 0.4802 | 0.4873 | 0.2830 | 0.2576 | 0.4134 | 0.4559 | 0.5061 | 0.0000 |
| R0_HGB | 4 | 0.5016 | 0.5021 | 0.3307 | 0.2525 | 0.4418 | 0.4745 | 0.5273 | 0.0000 |
| R1_substitute | 4 | 0.5133 | 0.5209 | 0.2620 | 0.2578 | 0.3777 | 0.4849 | 0.5416 | 0.0000 |
| B | 4 | 0.5583 | 0.5524 | 0.3229 | 0.2504 | 0.4274 | 0.5341 | 0.5811 | 0.0000 |
| B_S | 4 | 0.5572 | 0.5531 | 0.3243 | 0.2501 | 0.4250 | 0.5333 | 0.5798 | 0.0000 |
| B_bal | 4 | 0.5342 | 0.5317 | 0.3589 | 0.2511 | 0.3793 | 0.5108 | 0.5563 | 0.0000 |
| B_bal_S | 4 | 0.5409 | 0.5347 | 0.3619 | 0.2500 | 0.4065 | 0.5185 | 0.5624 | 0.0000 |
| B_A | 4 | 0.5602 | 0.5488 | 0.3274 | 0.2501 | 0.4042 | 0.5379 | 0.5811 | 0.0000 |
| B_A_S | 4 | 0.5593 | 0.5489 | 0.3296 | 0.2500 | 0.4070 | 0.5372 | 0.5801 | 0.0000 |

![主AUROC及配对区间](final/macro_auroc.png)

![来源AUROC热图](final/source_auroc_heatmap.png)

| source | model | eligible | auroc | auprc | brier | cal_brier | positive_rate | n_rows | n_pairs | seed_auroc_std | low | high | invalid_fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AZ ELN 2023 | B | True | 0.5382 | 0.7759 | 0.4046 | 0.2692 | 0.7694 | 607.0000 | 376.0000 | 0.0204 | 0.4776 | 0.5982 | 0.0000 |
| AZ ELN 2023 | B_A | True | 0.5718 | 0.7965 | 0.3930 | 0.2637 | 0.7694 | 607.0000 | 376.0000 | 0.0124 | 0.5138 | 0.6315 | 0.0000 |
| AZ ELN 2023 | B_A_S | True | 0.5688 | 0.7973 | 0.4027 | 0.2634 | 0.7694 | 607.0000 | 376.0000 | 0.0108 | 0.5120 | 0.6276 | 0.0000 |
| AZ ELN 2023 | B_S | True | 0.5446 | 0.7822 | 0.4070 | 0.2644 | 0.7694 | 607.0000 | 376.0000 | 0.0071 | 0.4852 | 0.6061 | 0.0000 |
| AZ ELN 2023 | B_bal | True | 0.5348 | 0.7824 | 0.4452 | 0.2669 | 0.7694 | 607.0000 | 376.0000 | 0.0160 | 0.4762 | 0.5952 | 0.0000 |
| AZ ELN 2023 | B_bal_S | True | 0.5360 | 0.7835 | 0.4374 | 0.2636 | 0.7694 | 607.0000 | 376.0000 | 0.0160 | 0.4766 | 0.5967 | 0.0000 |
| AZ ELN 2023 | R0_HGB | True | 0.5006 | 0.7875 | 0.4249 | 0.2703 | 0.7694 | 607.0000 | 376.0000 | 0.0000 | 0.4343 | 0.5665 | 0.0000 |
| AZ ELN 2023 | R0_RF | True | 0.5294 | 0.7715 | 0.3681 | 0.2852 | 0.7694 | 607.0000 | 376.0000 | 0.0072 | 0.4644 | 0.5971 | 0.0000 |
| AZ ELN 2023 | R1_substitute | True | 0.4648 | 0.7428 | 0.2945 | 0.2816 | 0.7694 | 607.0000 | 376.0000 | 0.0090 | 0.4008 | 0.5311 | 0.0000 |
| HiTEA 2024 | B | False | 0.6748 | 0.2809 | 0.1973 | 0.2283 | 0.1399 | 1387.0000 | 15.0000 | 0.0689 | NA | NA | NA |
| HiTEA 2024 | B_A | False | 0.6235 | 0.2419 | 0.2091 | 0.2357 | 0.1399 | 1387.0000 | 15.0000 | 0.0597 | NA | NA | NA |
| HiTEA 2024 | B_A_S | False | 0.6176 | 0.2417 | 0.2142 | 0.2361 | 0.1399 | 1387.0000 | 15.0000 | 0.0640 | NA | NA | NA |
| HiTEA 2024 | B_S | False | 0.6521 | 0.2508 | 0.2183 | 0.2342 | 0.1399 | 1387.0000 | 15.0000 | 0.0469 | NA | NA | NA |
| HiTEA 2024 | B_bal | False | 0.5984 | 0.2353 | 0.2796 | 0.2442 | 0.1399 | 1387.0000 | 15.0000 | 0.0785 | NA | NA | NA |
| HiTEA 2024 | B_bal_S | False | 0.5991 | 0.2357 | 0.2808 | 0.2442 | 0.1399 | 1387.0000 | 15.0000 | 0.0798 | NA | NA | NA |
| HiTEA 2024 | R0_HGB | False | 0.6844 | 0.3348 | 0.2077 | 0.2391 | 0.1399 | 1387.0000 | 15.0000 | 0.0000 | NA | NA | NA |
| HiTEA 2024 | R0_RF | False | 0.5829 | 0.2023 | 0.2257 | 0.2378 | 0.1399 | 1387.0000 | 15.0000 | 0.0018 | NA | NA | NA |
| HiTEA 2024 | R1_substitute | False | 0.6979 | 0.3389 | 0.2250 | 0.2486 | 0.1399 | 1387.0000 | 15.0000 | 0.0265 | NA | NA | NA |
| JNJ HTE 2024 | B | True | 0.4274 | 0.3230 | 0.3522 | 0.2492 | 0.3639 | 8016.0000 | 546.0000 | 0.0042 | 0.3815 | 0.4703 | 0.0000 |
| JNJ HTE 2024 | B_A | True | 0.4042 | 0.3180 | 0.3490 | 0.2499 | 0.3639 | 8016.0000 | 546.0000 | 0.0214 | 0.3589 | 0.4468 | 0.0000 |
| JNJ HTE 2024 | B_A_S | True | 0.4070 | 0.3172 | 0.3483 | 0.2493 | 0.3639 | 8016.0000 | 546.0000 | 0.0153 | 0.3611 | 0.4481 | 0.0000 |
| JNJ HTE 2024 | B_S | True | 0.4250 | 0.3218 | 0.3515 | 0.2493 | 0.3639 | 8016.0000 | 546.0000 | 0.0042 | 0.3788 | 0.4680 | 0.0000 |
| JNJ HTE 2024 | B_bal | True | 0.3793 | 0.3093 | 0.3524 | 0.2515 | 0.3639 | 8016.0000 | 546.0000 | 0.0126 | 0.3289 | 0.4250 | 0.0000 |
| JNJ HTE 2024 | B_bal_S | True | 0.4065 | 0.3196 | 0.3699 | 0.2506 | 0.3639 | 8016.0000 | 546.0000 | 0.0197 | 0.3614 | 0.4472 | 0.0000 |
| JNJ HTE 2024 | R0_HGB | True | 0.4418 | 0.3217 | 0.3223 | 0.2456 | 0.3639 | 8016.0000 | 546.0000 | 0.0000 | 0.3982 | 0.4850 | 0.0000 |
| JNJ HTE 2024 | R0_RF | True | 0.4556 | 0.3612 | 0.2389 | 0.2485 | 0.3639 | 8016.0000 | 546.0000 | 0.0192 | 0.4028 | 0.5143 | 0.0000 |
| JNJ HTE 2024 | R1_substitute | True | 0.3777 | 0.2992 | 0.2618 | 0.2504 | 0.3639 | 8016.0000 | 546.0000 | 0.0047 | 0.3184 | 0.4345 | 0.0000 |
| Merck 2025 | B | True | 0.5842 | 0.4825 | 0.2851 | 0.2450 | 0.4048 | 2668.0000 | 2668.0000 | 0.0448 | 0.5655 | 0.6035 | 0.0000 |
| Merck 2025 | B_A | True | 0.6146 | 0.4874 | 0.3097 | 0.2439 | 0.4048 | 2668.0000 | 2668.0000 | 0.0267 | 0.5960 | 0.6333 | 0.0000 |
| Merck 2025 | B_A_S | True | 0.6139 | 0.4896 | 0.3089 | 0.2438 | 0.4048 | 2668.0000 | 2668.0000 | 0.0270 | 0.5951 | 0.6326 | 0.0000 |
| Merck 2025 | B_S | True | 0.5808 | 0.4861 | 0.2823 | 0.2452 | 0.4048 | 2668.0000 | 2668.0000 | 0.0378 | 0.5622 | 0.5994 | 0.0000 |
| Merck 2025 | B_bal | True | 0.5763 | 0.4535 | 0.3514 | 0.2432 | 0.4048 | 2668.0000 | 2668.0000 | 0.0504 | 0.5584 | 0.5944 | 0.0000 |
| Merck 2025 | B_bal_S | True | 0.5756 | 0.4541 | 0.3506 | 0.2432 | 0.4048 | 2668.0000 | 2668.0000 | 0.0483 | 0.5576 | 0.5938 | 0.0000 |
| Merck 2025 | R0_HGB | True | 0.4768 | 0.3895 | 0.3053 | 0.2482 | 0.4048 | 2668.0000 | 2668.0000 | 0.0000 | 0.4547 | 0.4995 | 0.0000 |
| Merck 2025 | R0_RF | True | 0.4134 | 0.3473 | 0.2752 | 0.2477 | 0.4048 | 2668.0000 | 2668.0000 | 0.0071 | 0.3963 | 0.4305 | 0.0000 |
| Merck 2025 | R1_substitute | True | 0.5955 | 0.4826 | 0.2497 | 0.2499 | 0.4048 | 2668.0000 | 2668.0000 | 0.0130 | 0.5749 | 0.6158 | 0.0000 |
| Roche 2023 | B | True | 0.6832 | 0.6282 | 0.2497 | 0.2383 | 0.4488 | 1680.0000 | 65.0000 | 0.0259 | 0.6255 | 0.7372 | 0.0000 |
| Roche 2023 | B_A | True | 0.6504 | 0.5932 | 0.2579 | 0.2428 | 0.4488 | 1680.0000 | 65.0000 | 0.0326 | 0.6052 | 0.6947 | 0.0000 |
| Roche 2023 | B_A_S | True | 0.6476 | 0.5913 | 0.2584 | 0.2436 | 0.4488 | 1680.0000 | 65.0000 | 0.0372 | 0.6016 | 0.6923 | 0.0000 |
| Roche 2023 | B_S | True | 0.6783 | 0.6223 | 0.2565 | 0.2415 | 0.4488 | 1680.0000 | 65.0000 | 0.0295 | 0.6215 | 0.7323 | 0.0000 |
| Roche 2023 | B_bal | True | 0.6463 | 0.5815 | 0.2866 | 0.2428 | 0.4488 | 1680.0000 | 65.0000 | 0.0230 | 0.5972 | 0.6931 | 0.0000 |
| Roche 2023 | B_bal_S | True | 0.6456 | 0.5815 | 0.2895 | 0.2428 | 0.4488 | 1680.0000 | 65.0000 | 0.0255 | 0.5959 | 0.6928 | 0.0000 |
| Roche 2023 | R0_HGB | True | 0.5873 | 0.5097 | 0.2702 | 0.2460 | 0.4488 | 1680.0000 | 65.0000 | 0.0000 | 0.5187 | 0.6515 | 0.0000 |
| Roche 2023 | R0_RF | True | 0.5225 | 0.4692 | 0.2497 | 0.2489 | 0.4488 | 1680.0000 | 65.0000 | 0.0063 | 0.4794 | 0.5661 | 0.0000 |
| Roche 2023 | R1_substitute | True | 0.6152 | 0.5588 | 0.2422 | 0.2495 | 0.4488 | 1680.0000 | 65.0000 | 0.0137 | 0.5462 | 0.6842 | 0.0000 |

固定比较与消融（所有预注册对比统一Holm校正）：

| candidate | comparator | delta_auroc | low | high | valid_replicates | invalid_fraction | p_bootstrap_centered | p_method | p_holm |
|---|---|---|---|---|---|---|---|---|---|
| B_S | B | -0.0011 | -0.0068 | 0.0047 | 5000 | 0.0000 | 0.7087 | two-sided centered paired bootstrap; approximate | 1.0000 |
| B_S | B_bal | 0.0230 | 0.0076 | 0.0377 | 5000 | 0.0000 | 0.0028 | two-sided centered paired bootstrap; approximate | 0.0182 |
| B_S | R0_RF | 0.0769 | 0.0458 | 0.1057 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_S | R0_HGB | 0.0555 | 0.0254 | 0.0863 | 5000 | 0.0000 | 0.0004 | two-sided centered paired bootstrap; approximate | 0.0056 |
| B_S | R1_substitute | 0.0438 | 0.0163 | 0.0707 | 5000 | 0.0000 | 0.0016 | two-sided centered paired bootstrap; approximate | 0.0144 |
| B_A | B | 0.0020 | -0.0116 | 0.0160 | 5000 | 0.0000 | 0.7718 | two-sided centered paired bootstrap; approximate | 1.0000 |
| B_A | B_bal | 0.0260 | 0.0172 | 0.0352 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_A | R0_RF | 0.0800 | 0.0512 | 0.1073 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_A | R0_HGB | 0.0586 | 0.0301 | 0.0887 | 5000 | 0.0000 | 0.0006 | two-sided centered paired bootstrap; approximate | 0.0072 |
| B_A | R1_substitute | 0.0469 | 0.0185 | 0.0757 | 5000 | 0.0000 | 0.0022 | two-sided centered paired bootstrap; approximate | 0.0176 |
| B_A_S | B | 0.0010 | -0.0127 | 0.0153 | 5000 | 0.0000 | 0.8806 | two-sided centered paired bootstrap; approximate | 1.0000 |
| B_A_S | B_bal | 0.0251 | 0.0161 | 0.0346 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_A_S | R0_RF | 0.0791 | 0.0510 | 0.1060 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_A_S | R0_HGB | 0.0577 | 0.0293 | 0.0871 | 5000 | 0.0000 | 0.0006 | two-sided centered paired bootstrap; approximate | 0.0072 |
| B_A_S | R1_substitute | 0.0460 | 0.0170 | 0.0750 | 5000 | 0.0000 | 0.0026 | two-sided centered paired bootstrap; approximate | 0.0182 |
| B_A_S | B_A | -0.0009 | -0.0033 | 0.0013 | 5000 | 0.0000 | 0.4363 | two-sided centered paired bootstrap; approximate | 1.0000 |
| B_A_S | B_S | 0.0021 | -0.0123 | 0.0172 | 5000 | 0.0000 | 0.7730 | two-sided centered paired bootstrap; approximate | 1.0000 |
| B_A_S | B_bal_S | 0.0184 | 0.0096 | 0.0276 | 5000 | 0.0000 | 0.0004 | two-sided centered paired bootstrap; approximate | 0.0056 |
| B_bal_S | B_bal | 0.0067 | 0.0038 | 0.0096 | 5000 | 0.0000 | 0.0002 | two-sided centered paired bootstrap; approximate | 0.0040 |
| B_bal | B | -0.0241 | -0.0383 | -0.0092 | 5000 | 0.0000 | 0.0014 | two-sided centered paired bootstrap; approximate | 0.0140 |

预注册简化方案逐一判定，不挑选外层赢家后冒充新的独立确认：

| candidate | preregistered | family | meets_secondary_rule | minimum_source_delta | delta_vs_balanced_swa | interpretation |
|---|---|---|---|---|---|---|
| B_A | True | all frozen contrasts, Holm corrected | False | -0.0515 | 0.0193 | Secondary evidence only; no post-test winner declared as independent primary confirmation |
| B_S | True | all frozen contrasts, Holm corrected | False | -0.0306 | 0.0162 | Secondary evidence only; no post-test winner declared as independent primary confirmation |

## 不确定性、校准与覆盖

| model | metric | scheme | low | high | valid_replicates | invalid_fraction |
|---|---|---|---|---|---|---|
| R0_RF | auroc | pair | 0.4559 | 0.5061 | 5000 | 0.0000 |
| R0_RF | auroc | pigeonhole | 0.4345 | 0.5260 | 1000 | 0.0000 |
| R0_RF | auprc | pair | 0.4538 | 0.5267 | 5000 | 0.0000 |
| R0_RF | auprc | pigeonhole | 0.4239 | 0.5536 | 1000 | 0.0000 |
| R0_RF | brier | pair | 0.2778 | 0.2880 | 5000 | 0.0000 |
| R0_RF | brier | pigeonhole | 0.2691 | 0.2966 | 1000 | 0.0000 |
| R0_RF | worst_source_auroc | pair | 0.3945 | 0.4296 | 5000 | 0.0000 |
| R0_HGB | auroc | pair | 0.4745 | 0.5273 | 5000 | 0.0000 |
| R0_HGB | auroc | pigeonhole | 0.4520 | 0.5504 | 1000 | 0.0000 |
| R0_HGB | auprc | pair | 0.4724 | 0.5369 | 5000 | 0.0000 |
| R0_HGB | auprc | pigeonhole | 0.4407 | 0.5673 | 1000 | 0.0000 |
| R0_HGB | brier | pair | 0.3181 | 0.3441 | 5000 | 0.0000 |
| R0_HGB | brier | pigeonhole | 0.3005 | 0.3640 | 1000 | 0.0000 |
| R0_HGB | worst_source_auroc | pair | 0.3977 | 0.4753 | 5000 | 0.0000 |
| R1_substitute | auroc | pair | 0.4849 | 0.5416 | 5000 | 0.0000 |
| R1_substitute | auroc | pigeonhole | 0.4678 | 0.5642 | 1000 | 0.0000 |
| R1_substitute | auprc | pair | 0.4917 | 0.5546 | 5000 | 0.0000 |
| R1_substitute | auprc | pigeonhole | 0.4692 | 0.5834 | 1000 | 0.0000 |
| R1_substitute | brier | pair | 0.2589 | 0.2651 | 5000 | 0.0000 |
| R1_substitute | brier | pigeonhole | 0.2558 | 0.2684 | 1000 | 0.0000 |
| R1_substitute | worst_source_auroc | pair | 0.3184 | 0.4328 | 5000 | 0.0000 |
| B | auroc | pair | 0.5341 | 0.5811 | 5000 | 0.0000 |
| B | auroc | pigeonhole | 0.5129 | 0.6018 | 1000 | 0.0000 |
| B | auprc | pair | 0.5242 | 0.5825 | 5000 | 0.0000 |
| B | auprc | pigeonhole | 0.4982 | 0.6105 | 1000 | 0.0000 |
| B | brier | pair | 0.3119 | 0.3339 | 5000 | 0.0000 |
| B | brier | pigeonhole | 0.2970 | 0.3496 | 1000 | 0.0000 |
| B | worst_source_auroc | pair | 0.3815 | 0.4699 | 5000 | 0.0000 |
| B_S | auroc | pair | 0.5333 | 0.5798 | 5000 | 0.0000 |
| B_S | auroc | pigeonhole | 0.5121 | 0.6016 | 1000 | 0.0000 |
| B_S | auprc | pair | 0.5247 | 0.5830 | 5000 | 0.0000 |
| B_S | auprc | pigeonhole | 0.4978 | 0.6093 | 1000 | 0.0000 |
| B_S | brier | pair | 0.3131 | 0.3355 | 5000 | 0.0000 |
| B_S | brier | pigeonhole | 0.2985 | 0.3525 | 1000 | 0.0000 |
| B_S | worst_source_auroc | pair | 0.3788 | 0.4680 | 5000 | 0.0000 |
| B_bal | auroc | pair | 0.5108 | 0.5563 | 5000 | 0.0000 |
| B_bal | auroc | pigeonhole | 0.4897 | 0.5722 | 1000 | 0.0000 |
| B_bal | auprc | pair | 0.5046 | 0.5613 | 5000 | 0.0000 |
| B_bal | auprc | pigeonhole | 0.4790 | 0.5857 | 1000 | 0.0000 |
| B_bal | brier | pair | 0.3472 | 0.3709 | 5000 | 0.0000 |
| B_bal | brier | pigeonhole | 0.3311 | 0.3898 | 1000 | 0.0000 |
| B_bal | worst_source_auroc | pair | 0.3289 | 0.4250 | 5000 | 0.0000 |
| B_bal_S | auroc | pair | 0.5185 | 0.5624 | 5000 | 0.0000 |
| B_bal_S | auroc | pigeonhole | 0.4964 | 0.5785 | 1000 | 0.0000 |
| B_bal_S | auprc | pair | 0.5078 | 0.5637 | 5000 | 0.0000 |
| B_bal_S | auprc | pigeonhole | 0.4816 | 0.5871 | 1000 | 0.0000 |
| B_bal_S | brier | pair | 0.3496 | 0.3740 | 5000 | 0.0000 |
| B_bal_S | brier | pigeonhole | 0.3331 | 0.3933 | 1000 | 0.0000 |
| B_bal_S | worst_source_auroc | pair | 0.3614 | 0.4472 | 5000 | 0.0000 |
| B_A | auroc | pair | 0.5379 | 0.5811 | 5000 | 0.0000 |
| B_A | auroc | pigeonhole | 0.5162 | 0.5970 | 1000 | 0.0000 |
| B_A | auprc | pair | 0.5225 | 0.5775 | 5000 | 0.0000 |
| B_A | auprc | pigeonhole | 0.4981 | 0.5997 | 1000 | 0.0000 |
| B_A | brier | pair | 0.3170 | 0.3381 | 5000 | 0.0000 |
| B_A | brier | pigeonhole | 0.3008 | 0.3561 | 1000 | 0.0000 |
| B_A | worst_source_auroc | pair | 0.3589 | 0.4468 | 5000 | 0.0000 |
| B_A_S | auroc | pair | 0.5372 | 0.5801 | 5000 | 0.0000 |
| B_A_S | auroc | pigeonhole | 0.5145 | 0.5969 | 1000 | 0.0000 |
| B_A_S | auprc | pair | 0.5226 | 0.5772 | 5000 | 0.0000 |
| B_A_S | auprc | pigeonhole | 0.4984 | 0.6003 | 1000 | 0.0000 |
| B_A_S | brier | pair | 0.3191 | 0.3404 | 5000 | 0.0000 |
| B_A_S | brier | pigeonhole | 0.3027 | 0.3602 | 1000 | 0.0000 |
| B_A_S | worst_source_auroc | pair | 0.3611 | 0.4481 | 5000 | 0.0000 |

主区间使用来源内底物对分组配对bootstrap，所有方法和seed共享抽样；seed不是独立化学实验。双向芳基/胺pigeonhole为共享底物敏感性。只有4个主来源，区间条件于这些来源及已拟合轨迹，不能覆盖所有新实验室或训练集不确定性。Merck严格测试虽含许多底物对，却仅含8个胺实体，双向bootstrap不能据此创造更丰富的独立化学覆盖。

逐来源原始/校准可靠性数据与图见final/reliability_by_source_seed.csv及reliability_source_*.png。Top-k仅来自同对多个实测条件，重复条件对整体排除并报告覆盖；见final/topk_measured_conditions.csv或topk_status.json。

结构相似度固定bins为[0,.3],(.3,.5],(.5,.7],(.7,1]；粗类型/试剂覆盖结果均为探索性描述，不进行选择性分层机理宣称：

| source | n_rows | n_pairs | n_aryls | n_amines | aryl_similarity_mean | amine_similarity_mean | catalyst_exact_coverage | base_exact_coverage | solvent_exact_coverage | positive_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| AZ ELN 2023 | 607 | 376 | 230 | 207 | 0.3789 | 0.3952 | 0.0000 | 0.9687 | 0.9638 | 0.7694 |
| HiTEA 2024 | 1387 | 15 | 14 | 13 | 0.3729 | 0.5458 | 0.0303 | 0.4261 | 0.8810 | 0.1399 |
| JNJ HTE 2024 | 8016 | 546 | 90 | 65 | 0.4272 | 0.3419 | 0.0195 | 0.8977 | 1.0000 | 0.3639 |
| Merck 2025 | 2668 | 2668 | 335 | 8 | 0.4059 | 0.4205 | 0.0000 | 0.0000 | 1.0000 | 0.4048 |
| Roche 2023 | 1680 | 65 | 28 | 18 | 0.4704 | 0.4321 | 0.0000 | 1.0000 | 1.0000 | 0.4488 |

精确试剂覆盖保留电荷、盐和配位形态。AZ/Merck/Roche催化剂精确结构训练覆盖可能为0，JNJ约1.95%；这不等于催化剂化学类别全新。需同时考虑底物新颖性、相似度、试剂形态覆盖和来源正类率。全部分层含行/底物对/来源覆盖见final/exploratory_domain_strata.csv。

主实验逐来源退步、消融、校准和实测条件覆盖的解释见 [主实验失败分析](PRIMARY_FAILURE_ANALYSIS.md)。

## 预注册敏感性分析

### author_labels

| model | n_sources | auroc | auprc | brier | cal_brier | worst_source_auroc |
|---|---|---|---|---|---|---|
| R1_substitute | 4 | 0.5153 | 0.5297 | 0.2647 | 0.2647 | 0.3766 |
| B | 4 | 0.5512 | 0.5568 | 0.3288 | 0.3288 | 0.4251 |
| B_S | 4 | 0.5517 | 0.5578 | 0.3291 | 0.3291 | 0.4223 |
| B_A | 4 | 0.5579 | 0.5525 | 0.3308 | 0.3308 | 0.4083 |
| B_A_S | 4 | 0.5571 | 0.5532 | 0.3334 | 0.3334 | 0.4116 |

共同测试行的点差为敏感性模型减主模型，使用原始概率。每个 truth_regime 下，两套模型均按同一套真值重算。表中先对 seed 平均，再对共同测试行覆盖的来源等权平均。任一来源单类时保留 NA。主宏指标见主表。

| model | truth_regime | auprc | auroc | brier |
|---|---|---|---|---|
| B | fixed_primary_truth | 0.0014 | -0.0015 | 0.0034 |
| B | native_sensitivity_truth | 0.0014 | -0.0015 | 0.0034 |
| B_A | fixed_primary_truth | 0.0025 | 0.0063 | -0.0020 |
| B_A | native_sensitivity_truth | 0.0025 | 0.0063 | -0.0020 |
| B_A_S | fixed_primary_truth | 0.0025 | 0.0070 | -0.0014 |
| B_A_S | native_sensitivity_truth | 0.0025 | 0.0070 | -0.0014 |
| B_S | fixed_primary_truth | 0.0015 | -0.0002 | 0.0008 |
| B_S | native_sensitivity_truth | 0.0015 | -0.0002 | 0.0008 |
| R1_substitute | fixed_primary_truth | 0.0042 | 0.0057 | 0.0020 |
| R1_substitute | native_sensitivity_truth | 0.0042 | 0.0057 | 0.0020 |

### dd_original

状态：PENDING。未读取未完成的预测，也不将部分矩阵作为完成证据。

Dreher Doyle原版本缺少具体添加剂身份，仅作为条件不完整敏感性，不纳入主结论。其加入会改变严格测试集合；共同row_id配对结果使用固定主真值，不能把集合变化解释成训练改善。作者标签敏感性同时报告原生标签与固定主真值，标签标准变化和模型变化不能混为一谈。

## 运行成本、失败与独立核验

实际训练审计：`{"status": "PASS", "formal_trajectories": 45, "matched_update_budgets": true, "resumable_states_verified": true, "all_swa_multiple_snapshots": true, "source_sha256": "5297bab9e3de4fbc6cb1089fa591126cd464a68f02f8e8330f95a1b3a10e9d0b"}`。逐轨迹更新数、样本曝光、SWA快照数及GroupDRO权重/有效样本量见final/training_budget_audit.csv与final/groupdro_trajectories.csv；权重图见final/groupdro_weight_trajectories.png。权重集中仅描述优化行为，不能证明未见来源泛化或反应机理。

预算记录：`{"gpu_hours": 23.419250863954012, "cpu_core_hours": 26.21638431, "active_cpu_threads": 1, "development_gpu_hours": 12.129947342723607, "feasibility_gpu_hours": 0.025113719089163673}`。失败与重试累计；upper_bound标记保守收费，包括结束时刻不明的孤儿任务及协调器自身CPU用量预留；子训练任务单独计费。完整成功/失败账本见final/run_inventory.csv及../runs/ledger.jsonl。报告生成任务若仍在运行，其最终小额成本以随后完成的guard账本为准。

账本无失败或中断的任务记录；环境下载中断见 [DECISIONS](../DECISIONS.md)。

CPU对照实际拟合证据：[30次主内层拟合审计](ALL_TREE_INNER_RUNTIME_AUDIT.json)、[45次主正式拟合审计](ALL_FORMAL_TREE_RUNTIME_AUDIT.json)、[30次敏感性RF拟合审计](ALL_SENSITIVITY_TREE_RUNTIME_AUDIT.json)。具体检查范围、训练分区、固定参数和已保存文件的核验记录见各审计报告；审计过程未读取外层预测或指标。

author_labels 分支的全部神经训练状态检查见 [审计记录](AUTHOR_LABELS_NEURAL_RUNTIME_AUDIT.json)。

独立审计PASS：逐条标签/来源/底物实体/底物对/条件ID与冻结records一致；锁定模型×seed×行矩阵完整；sklearn独立重建主表及逐来源逐seed指标，与原评价表在1e-10容差内一致。审计清单和源码hash见final/independent_audit.json。

## 局限、文献差异与复现

CAD、isolated yield、MisER与LCAP混合测量，统一10%阈值不消除测量差异；温度、时间、浓度与部分载量缺失。无标签预训练与测试结构的重叠不能排除。来源均衡可能解释部分收益，因此保留B_bal及B_bal_S。GroupDRO/SWA为成熟方法，注意力或特征归因不能证明机理。负结果如实交付，未达到门槛不强行宣称增量。

原文与直接近例已包含多来源或双底物OOD研究，差异来自受控实验而非首创这些任务。详见 [文献差异](LITERATURE_DIFFERENCES.md) 与 [期刊范围及subscription非OA路径](JOURNAL_SCOPE.md)。Reaction Chemistry & Engineering与Chemical Papers是否接受由编辑和评审判断，本项目不投稿。

```bash
# 已冻结项目；先恢复/完成协调器，避免重复启动已运行任务
.venv/bin/python scripts/pipeline.py --phase all
.venv/bin/python -m bh_transfer.evaluate --predictions artifacts/predictions.parquet --lock FINAL_EVAL_LOCK.json --out reports/final
.venv/bin/python scripts/sensitivity_pipeline.py --kind all --phase all
# 每个完整敏感性目录使用其自身predictions/lock运行同一evaluate模块
.venv/bin/python scripts/finalize_report.py --project .
```

请使用scripts/run_guard.py为实际CPU/GPU任务登记run_id、线程槽、成本与日志；上列命令展示入口，不授权并行GPU运行或改动冻结配置。
