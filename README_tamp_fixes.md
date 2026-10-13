# VLM-TAMP 修复迁移与测试

将 `../VLM-TAMP` 的 `2c052e4d8aee2183a82f7b66bd190b2c8c96372a` 中与规划有关的补丁三方合并到本仓库；保留本地 GPT/Qwen 接口、路径和 TAMP profile。没有合并作者 GPT55 provider，也没有替换 PDDLStream/Fast Downward。

## 已迁移

- grounding 的对象表示归一化、picked/sprinkledto 静态条件检查。
- reducer 保留 tuple 目标、处理无关动态事实。
- reduced-world 重新绑定 IR/IK/pull stream，并传递实际 IR 设置。
- `Containble` 拼写修正，picked 已完成判断，in/sprinkledto 目标区域处理。
- 抓取目标、附着物、支撑物的接触处理；修复底盘规划固定查询 body 1 的问题。
- 作者的锅、锅盖、调料、水龙头/炉灶的抓取、放置、优先底盘候选和水龙头可逆开度策略。
- approach 方向、鸡腿抓取数据库、双把手抓取读取。
- 空序列、加载状态路径、进度统计修复。

## 本地适配

- 双臂情况下，另一只手空闲时，不因单手持物而阻止继续 pick。
- 炉灶右旋钮优先按 `oven/stove` 的 `knob_joint_2` 名称匹配，兼容本地 `(6, 5)`；无名称的旧诊断 fixture 保留 `(6, 4)` 回退。
- 接受模型返回的单行 Python/JSON 字符串列表，避免将多个谓词视为一条子目标。
- 默认配置开启 `log_failures`、`use_all_grasps`；关闭 viewer，主测试入口使用 `use_learned_ir=False`、`serve_page=False`。
- 没有增加规划预算，没有关闭全局碰撞，没有开启 teleport。
- 显式传入 seed 不再强制开启 viewer；保留配置或命令行指定的显示设置。
- 固定序列模式保留跳过的失败目标在统计分母中，避免出现 `17/16`、进度超过 100%。`task_progress` 表示推进到的位置，仍应以 `task_success` 和逐目标状态判定成功。
- 每次求解结束后，将 PDDLStream 的 `visualizations/log.json` 复制到该实验的 `log/stream_failures/`，防止下一次求解覆盖。修复日志开关被一次关闭后无法重新开启的问题。未生成 stream 日志的求解不产生该 JSON；完整过程另见 launcher 和 `log/content/`。

这些不是全部通用算法修复：优先底盘坐标、水龙头开度、撒料高度仍包含作者厨房场景的经验参数；接触过滤在部分场景按整个 body 排除障碍。成功统计不能替代轨迹和接触的检查。除右旋钮外，其他场景专用编号仍需在更换模型/布局时重新验证。

## 运行

在 kitchen-worlds 根目录：

```bash
source /usr/local/miniconda/etc/profile.d/conda.sh
conda activate kitchen
source ./your_project_folder/gpt_api.sh
export PYTHONPATH="$PWD/pddlstream:$PWD/pybullet_planning:$PWD/lisdf:$PWD/pybullet_planning/motion:$PWD/..${PYTHONPATH:+:$PYTHONPATH}"
export TAMP_PROFILE=1
unset TAMP_CPROFILE
python pybullet_planning/tutorials/test_vlm_tamp.py \
  --vlm_type gpt4v \
  --open_goal "make chicken soup" \
  --dual_arm \
  --planning_mode sequence-reprompt \
  --exp_subdir test_gpt4o_mini_Nrp_2_dual_arm_try_fix \
  --log_failures
```

不固定 seed，不加载作者 memory，使用用户指定的 fresh GPT 输出。实际 seed 和参数在运行日志和实验文件中保存。

输出位于 `experiments/test_gpt4o_mini_Nrp_2_dual_arm_try_fix/`。本次代理运行的启动日志为 `launcher_kitchen.log`；`launcher.log` 是首次沙箱禁止联网导致的失败记录，`launcher_network_retry.log` 是 Graphviz 未加入 PATH 导致的失败记录。这两次均发生在 TAMP 开始之前。需激活完整 kitchen 环境，而不只是指定其 Python 可执行文件，以便找到 Graphviz dot。

## 验证

迁入的回归测试及本地双臂/单行列表/旋钮编号/失败日志/固定序列统计测试共 74 项通过；现有 profile 测试 6 项通过。真实整任务运行结果需结合 `time.json`、`vlm-tamp.csv` 及失败 stream 日志判断，不能由单元测试推断成功。

## 固定 memory 对照

使用上面的环境设置，先按作者单臂、easy 和 seed 测试。`sequence` 的 reprompt 上限为 0；加载的是模型回答缓存，仍需重新进行 TAMP 求解和动作执行。

```bash
python pybullet_planning/tutorials/test_vlm_tamp.py \
  --seed 163381 --vlm_type gpt4v --open_goal "make chicken soup" \
  --planning_mode sequence \
  --load_llm_memory "$PWD/../VLM-TAMP/artifacts/gpt55_chicken_soup_success" \
  --exp_subdir test_author_memory_try_fix --log_failures
```

后续完整 VLM+TAMP 测试移除 `--load_llm_memory`，使用 `--dual_arm --planning_mode sequence-reprompt`；源码中该模式最多触发 2 次 reprompt。

```bash
python pybullet_planning/tutorials/test_vlm_tamp.py \
  --seed 163381 --vlm_type gpt4v --open_goal "make chicken soup" \
  --dual_arm --planning_mode sequence-reprompt \
  --exp_subdir test_gpt4o_mini_Nrp_2_dual_arm_fixed --log_failures
```

## 固定 memory 结果与水槽诊断

- 完整基线：`test_author_memory_try_fix/261013_012431_vlm-tamp`，`task_success=0`，规划合计 309.34 s。原始 17 个子目标中 16 个通过（包含 1 个 already），第 7 步锅从水槽返回台面失败；其余失败尝试后来重试成功。该轮统计修复前显示 `17/16`，这是进度分母错误，不是 task success。
- 第 7 步三次重试耗时 1.4958 / 1.3371 / 14.3844 s。失败状态保存在 `states/agent_state_6*.pkl`，逐次失败记录在 `log/stream_failures/problem_6_*.json`。
- 对作者成功底盘站位进行相对坐标迁移后，独立 pick probe 的两个抓取姿态可行，但完整 replan 仍因 approach path 和放置 IK 等失败，不能由 probe 推断整个子目标可行。脚本与输出保存在 `test_author_memory_try_fix/diagnostics/`。
- 候选对照：`test_author_memory_sink_fix/261013_013155_vlm-tamp`，同 seed/memory，加入水槽优先候选。第 7 步三次均失败，耗时 33.9195 / 41.7601 / 49.7767 s，合计 125.4563 s。该轮在确认候选无效后主动中止（exit 130），不是完整实验；候选改动已从生产代码撤回，不属于当前修复。该轮已包含固定序列统计修复。
- 完整固定 memory 运行的原始文件保持不变；后续统计以逐目标状态及 `task_success` 为准。

## 最终 fresh VLM+TAMP 测试

实验：`experiments/test_gpt4o_mini_Nrp_2_dual_arm_fixed/261013_013729_vlm-tamp`。

- 已 source 用户的 `your_project_folder/gpt_api.sh`，实际调用日志显示 `GPT-4v (gpt-4o-mini)`。
- seed=163381，双臂，easy，`load_memory=None`，`planning_mode=sequence-reprompt`，`max_reprompts=2`。
- 正常退出（exit 0），`task_success=1`，10/10 子目标通过，7 次求解全部成功，另外 3 步已满足；本轮实际 reprompt 次数为 0。
- TAMP 规划累计 130.32 s，18 个规划动作；两次新 VLM 请求累计约 6.59 s。二者不等同于整个进程墙钟时间，后者还包含建场景、执行、渲染和日志。
- 7 份 profile 位于 `tamp_profiles/`；7 份逐次 stream 日志位于 `log/stream_failures/`，这些日志也可包含最终成功求解中的失败采样。完整终端日志为上一级 `launcher.log`，实验内另有 `log/content/`、状态与命令文件。

**成功口径限制**：模型生成的 10 步只有开冰箱门、抓鸡肉、放到锅底、开柜门、抓取并撒两种调料、盖锅盖。它没有生成把锅移到水槽、开关水龙头、把锅移到炉灶和开炉灶的目标。因此该轮的 `task_success=1` 是仓库对生成子目标序列的成功判定，不能作为作者 17 步完整做汤流程已复现成功的证据。作者固定序列的第 7 步仍是待解决的卡点。

## 已结束的首次 fresh VLM 测试

`test_gpt4o_mini_Nrp_2_dual_arm_try_fix/261013_011727_vlm-tamp`：seed=552261，双臂，2 次 reprompt，`task_success=0`，进度 6/13，总规划时间 314.38 s。鸡肉、盐和胡椒的 pick 成功；冰箱门、左柜门、右柜门的 close 分别耗尽该分支的重试。该轮在逐次 stream JSON 归档补丁前运行，详细失败信息见 `launcher_kitchen.log` 和实验 `log/content/`，不能声称有完整的逐次失败 JSON。
