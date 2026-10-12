# TAMP 耗时 profile

在现有能正常运行的 kitchen 环境和启动命令上增加环境变量即可：

```bash
export TAMP_PROFILE=1
# 然后执行原来的 test_vlm_tamp.py 或 run_vlm_paper.py 命令
```

默认关闭；`unset TAMP_PROFILE` 关闭。不要因此更改 seed、双臂、difficulty、超时等实验设置。保持原有无显示配置；本功能不增加 `--no-viewer` 参数。

可选变量：

```bash
export TAMP_CPROFILE=1             # 额外输出 Python cProfile，仅在 TAMP_PROFILE 开启时生效
export TAMP_PROFILE_DIR=/some/path # 可选；默认使用每次实验 self.exp_dir
```

首轮建议只设置 TAMP_PROFILE。cProfile 会增加明显开销，尤其是在有大量 Python 调用时，可另跑一轮找调用热点。

## 输出

每次最外层 agent.replan 完成（包括正常无解、提前返回、可传播异常）保存：

```text
experiments/<实验组>/<运行目录>/tamp_profiles/<时间戳>_<唯一ID>.json
experiments/<实验组>/<运行目录>/tamp_profiles/<时间戳>_<唯一ID>.pstats  # 可选
```

每个文件是一次 replan 尝试；同一子目标换 object reducer 再次求解会有另一个文件。metadata 保存 goal、node、reducer_index、状态、PID。结合相同运行目录的 planning_config.json/time.json 查看模型、seed 和子目标结果。

TAMP 内部捕获并处理的超时通常表现为 no_plan；异常传播到 replan 外时记录 exception。强制 SIGKILL、进程崩溃或断电无法保证当前尝试落盘，已完成尝试不受影响。结束 replan 前文件尚不会出现。

## 三个层次

从 schema_version=2 开始，JSON 优先展示 `call_tree`：按实际插桩调用路径，从外到内嵌套 `children`，每个节点显示 calls、inclusive_s、exclusive_s、max_s、exceptions 和 mean_s。兄弟节点按首次进入顺序排列；同一父路径下的同名调用合并，因此这是聚合调用树，不是逐次调用的时间线。同名 solver 在 grounding 和正式规划下面分别统计。每个节点满足 inclusive_s ≈ exclusive_s + 直接 children 的 inclusive_s 之和，不能跨层重复相加。未插桩的函数不会出现在树上。

原来的 `spans`、`stream_events` 和 `kernels_by_stream` 仍保留。旧 JSON 未保存完整调用路径，无法精确恢复此树；重新开启 profile 运行后，新文件自动包含 call_tree。

1. `spans`：按阶段/函数名聚合。包含 calls、inclusive_s、exclusive_s、mean_s、max_s、exceptions。
2. `stream_events`：每次 stream 求值的相对起始时间、耗时、成功/异常，用于分析长尾。stream 聚合另含 successful_calls、no_success_calls、p50_s、p95_s（nearest rank）。无新成功结果不一定是碰撞失败，也可能耗尽候选、重复结果或其他不可行原因。
3. `kernels_by_stream`：每种 stream 内部发生的 IK、碰撞、状态更新和路径规划的调用次数与耗时。没有 stream 父调用时归为 outside_stream。

| 名称 | 计时边界 |
|---|---|
| attempt.total | 最外层 replan，包括观测图片、日志、问题缩减和后处理；不等同于纯求解器 |
| phase.observation_logging / result_logging | LLAMP replan 的前后处理 |
| phase.object_reduction | 规划对象缩减 |
| phase.grounding | 子目标 grounding 检查；可能包含自己的求解调用 |
| phase.solve_wrapper | solve_pddlstream 整体 |
| phase.solver | solve_one，包括调用 focused |
| phase.cleanup_and_knowledge | 恢复场景、清理临时物体、重新解析 knowledge |
| phase.preimage | 计划 preimage 提取及相关后处理 |
| solver.parse | PDDLStream problem 解析 |
| solver.focused | focused 主求解函数，所有早退路径均计时 |
| solver.sequencing | iterative_plan_streams，含符号规划及相关调度 |
| solver.instantiate / sas_translate | 符号实例化 / SAS 转换 |
| solver.fd_search_wrapper | FD 搜索入口，包含文件和结果处理 |
| solver.fd_subprocess | Popen + communicate 的墙钟时间，含子进程启动和等待，不是 FD CPU time |
| solver.refinement / refinement_process / refinement_queue | 连续求值的几个控制入口，可能相互嵌套 |
| stream.<名字> | 实际 next_results 求值，包括位姿、抓取、IR、IK、路径和碰撞 test 等 |
| kernel.ik.ikfast_solver | 调用 IKFast 解析求解的包装函数 |
| kernel.ik.ikfast_candidates | IKFast 候选生成器，每次恢复运行的时间，不含 yield 暂停 |
| kernel.ik.tracik / pybullet / approach | 对应后端求解包装 / approach IK |
| kernel.collision.closest_points | 包装 getClosestPoints 的函数，含参数处理和结果转换；不是 C++ 内部单独计时 |
| kernel.collision.pairwise / configuration | 物体对 / 配置可行性检查（后者还包含限位、状态设置） |
| kernel.state.set_joints / set_pose | 设置关节和物体位姿的包装；不能解读为纯 FK 时间 |
| kernel.motion.joint / waypoints | 关节路径规划 / 插值路径检查，底盘 stream 中的路径计算也可能调用这些函数 |

这里只覆盖 instrumented 调用路径。第三方代码直接调用 PyBullet、其他 IK 后端或不同运动规划入口时，时间仍包含在父 stream 中，但不会自动获得 kernel 标签。TracIK 模块只有原本使用时才加载，不会因为 profile 新增 TracIK 依赖。

`inclusive_s` 包含子调用，禁止跨层直接相加。例如 IK stream 已经包含其碰撞耗时。`exclusive_s` 排除所有直接嵌套的已插桩 span；未插桩计算仍留在父函数自身时间中。它不是严格的 Python self CPU time，也不能把所有剩余时间解释成 Python 开销。递归/重复同名 span 的 inclusive 时间也可能重叠。

sampling 没有伪造为独立互斥阶段：抓取/位姿等 stream 自身也可能进行碰撞检查，可用该 stream 的 inclusive/exclusive 时间及 kernels_by_stream 分析。未单独插桩完整 FK 和 PyBullet C++ 内部阶段。

实际动作执行发生在 replan 之外，本次文件不统计执行时间；普通 VLM 请求也不在此边界中。不得把这些文件的总时间当作完整 VLM+TAMP 实验墙钟时间。

## 阅读结果

```python
import json
from pathlib import Path
for path in sorted(Path('/实际实验目录/tamp_profiles').glob('*.json')):
    data = json.loads(path.read_text())
    print(path.name, data['metadata'])
    for name, s in sorted(data['spans'].items(), key=lambda kv: kv[1]['exclusive_s'], reverse=True):
        print(name, 'calls=', s['calls'], 'inclusive=', round(s['inclusive_s'], 4),
              'exclusive=', round(s['exclusive_s'], 4))
```

`python -m pstats /实际路径/文件.pstats` 可以查看 cProfile。它能看到调用 FD 的等待，但看不到 FD 子进程内部的函数调用。

比较加速效果时固定场景、memory、seed、规划参数，并同时报告成功率、完成子目标、stream 调用次数和耗时。计时开销可能影响墙钟超时判定，因此开启 profile 不保证与未插桩运行获得完全相同结果。先使用少量可重复尝试测量开销，再进行多次运行比较。

## 验证

在 kitchen-worlds 目录：

```bash
PYTHONPATH="$PWD/pddlstream:$PYTHONPATH" python -m unittest discover -s tests -p test_tamp_profile.py -v
```

测试覆盖嵌套计时、stream 归因、生成器 yield/send/返回值、关闭时透明调用、异常落盘、嵌套 replan 只输出一份及可选 pstats。
