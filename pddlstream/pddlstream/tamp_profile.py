"""Opt-in, nested wall-clock profiling. No robotics imports or random draws.

TAMP_PROFILE=1 enables per-replan JSON; TAMP_CPROFILE=1 adds Python pstats.
Exclusive seconds exclude directly nested instrumented spans, not all Python calls.
"""
import contextvars
import functools
import inspect
import json
import os
from pathlib import Path
import time
import uuid
from contextlib import contextmanager

_ACTIVE = contextvars.ContextVar("tamp_profile", default=None)


def enabled():
    return os.environ.get("TAMP_PROFILE", "").lower() in ("1", "true", "yes")


class Recorder:
    def __init__(self, metadata):
        self.metadata = metadata
        self.stats = {}
        self.stack = []
        self.events = []
        self.kernels_by_stream = {}
        self.call_tree = {}
        self.started = time.perf_counter()

    @contextmanager
    def span(self, name):
        parent = self.stack[-1][0] if self.stack else None
        siblings = self.stack[-1][3]['children'] if self.stack else self.call_tree
        if name not in siblings:
            siblings[name] = dict(name=name, calls=0, inclusive_s=0.0,
                                  exclusive_s=0.0, max_s=0.0, exceptions=0, children={})
        node = siblings[name]
        # 第四项记录当前调用路径的树节点；同名函数在不同父路径下分别统计。
        frame = [name, time.perf_counter(), 0.0, node]
        self.stack.append(frame)
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally: #退出 （无论正常结束还是抛异常，finally 都会执行）
            elapsed = time.perf_counter() - frame[1] #elapsed是这个span的总耗时
            self.stack.pop()  # 把自己从栈里弹出，再把 elapsed 加到父 span 的"子 span 累计耗时"上
            if self.stack:
                self.stack[-1][2] += elapsed
            stat = self.stats.setdefault(name, dict(calls=0, inclusive_s=0.0,
                exclusive_s=0.0, max_s=0.0, exceptions=0))
            #更新stats[name]
            stat['calls'] += 1 #调用了几次
            stat['inclusive_s'] += elapsed #总耗时（包含里面嵌套的子span）
            stat['exclusive_s'] += max(0.0, elapsed - frame[2])  #自身耗时，等于总耗时减去直接子span的耗时
            stat['max_s'] = max(stat['max_s'], elapsed)  #单次调用的最长耗时
            stat['exceptions'] += error is not None  # 有几次是以异常结束的
            node['calls'] += 1
            node['inclusive_s'] += elapsed
            node['exclusive_s'] += max(0.0, elapsed - frame[2])
            node['max_s'] = max(node['max_s'], elapsed)
            node['exceptions'] += error is not None
            if name.startswith('kernel.'):
                caller = next((f[0] for f in reversed(self.stack) if f[0].startswith('stream.')), 'outside_stream')
                item = self.kernels_by_stream.setdefault(caller, {}).setdefault(name,
                    dict(calls=0, inclusive_s=0.0, exclusive_s=0.0))
                item['calls'] += 1
                item['inclusive_s'] += elapsed
                item['exclusive_s'] += max(0.0, elapsed-frame[2])
            if name.startswith('stream.'):
                self.events.append(dict(name=name, parent=parent,
                    start_s=frame[1]-self.started, duration_s=elapsed,
                    exception=error))

    def export(self): # 导出成字典
        for stat in self.stats.values():
            stat['mean_s'] = stat['inclusive_s'] / stat['calls']
        for name, stat in self.stats.items():
            if name.startswith('stream.'): #对每种 stream，根据 events 里所有调用的耗时计算中位数 p50_s 和 95 分位数 p95_s，用的是最近秩法
                samples = sorted(e['duration_s'] for e in self.events if e['name'] == name)
                import math
                stat['p50_s'] = samples[max(0, math.ceil(len(samples)*0.5)-1)]
                stat['p95_s'] = samples[max(0, math.ceil(len(samples)*0.95)-1)]
        def export_tree(nodes):
            return [dict(node, mean_s=node['inclusive_s'] / node['calls'],
                         children=export_tree(node['children']))
                    for node in nodes.values()]

        return dict(schema_version=2, metadata=self.metadata,
                    elapsed_s=time.perf_counter()-self.started,
                    call_tree=export_tree(self.call_tree),
                    call_tree_semantics='aggregated by full instrumented call path; siblings ordered by first entry, not a chronological trace',
                    spans=self.stats,
                    stream_events=self.events, kernels_by_stream=self.kernels_by_stream,
                    accounting='inclusive spans overlap; exclusive spans exclude instrumented children only')


def timed(name):#给函数加计时
    """Decorate functions, or time only generator resumes (never suspension)."""
    def decorate(fn):
        if inspect.isgeneratorfunction(fn):
            @functools.wraps(fn)
            def generator(*args, **kwargs): # 生成器版本的包装
                iterator = fn(*args, **kwargs)
                if _ACTIVE.get() is None:
                    return (yield from iterator)
                value, pending = None, None
                try:
                    while True:
                        with span(name):
                            try:
                                if pending is None:
                                    item = iterator.send(value)
                                else:
                                    error, pending = pending, None
                                    item = iterator.throw(error)
                            except StopIteration as stop:
                                return stop.value
                        try:
                            value = yield item
                        except GeneratorExit:
                            raise
                        except BaseException as exc:
                            pending = exc
                finally:
                    iterator.close()
            return generator
        @functools.wraps(fn)  
        def wrapped(*args, **kwargs):  #普通函数版本的包装
            recorder = _ACTIVE.get()
            if recorder is None:   # 没开 profile，或者不在某次 replan 内部
                return fn(*args, **kwargs)   # 直接调用原函数，不计时
            with recorder.span(name):   # 开始计时
                return fn(*args, **kwargs) # 调用原函数；返回或抛异常时，with 都会停止计时
        return wrapped
    return decorate


@contextmanager
def span(name): #Recorder.span 的简单包装：有记录器就计时，没有就什么也不做。用在函数内部的一段代码上
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
    else:
        with recorder.span(name):
            yield


def stream_call(fn): #给stream求值计时
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        recorder = _ACTIVE.get()
        if recorder is None:
            return fn(self, *args, **kwargs)
        name = 'stream.' + self.external.name
        with recorder.span(name):
            result = fn(self, *args, **kwargs)
        success = any(r.is_successful() for r in result[0])
        stat = recorder.stats[name]
        field = 'successful_calls' if success else 'no_success_calls'
        stat[field] = stat.get(field, 0) + 1
        recorder.events[-1]['success'] = success
        return result
    return wrapped


def attempt(fn): # 每次replan生成一个文件
    """Outermost agent.replan owns the file; nested superclass calls share it."""
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not enabled() or _ACTIVE.get() is not None:
            return fn(self, *args, **kwargs)
        output = Path(os.environ.get('TAMP_PROFILE_DIR') or self.exp_dir) / 'tamp_profiles'
        output.mkdir(parents=True, exist_ok=True)
        ident = '%s_%s' % (time.strftime('%Y%m%d_%H%M%S'), uuid.uuid4().hex[:12])
        problem = kwargs.get('pddlstream_problem') or getattr(self, 'pddlstream_problem', None)
        metadata = dict(attempt_id=ident, run_dir=str(self.exp_dir),
                        goal=repr(getattr(problem, 'goal', None)),
                        node=repr(getattr(self, 'last_node', None)),
                        reducer_index=getattr(self, 'object_reducer_state', None),
                        pid=os.getpid(), status='running')
        recorder = Recorder(metadata)
        token = _ACTIVE.set(recorder)
        profiler = None
        if os.environ.get('TAMP_CPROFILE', '').lower() in ('1', 'true', 'yes'):
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        try:
            with recorder.span('attempt.total'):
                result = fn(self, *args, **kwargs)
            metadata['status'] = 'solved' if getattr(self, 'plan', None) is not None else 'no_plan'
            return result
        except BaseException as exc:
            metadata['status'] = 'exception'
            metadata['exception'] = type(exc).__name__
            raise
        finally:
            if profiler is not None:
                profiler.disable()
            _ACTIVE.reset(token)
            # Never replace a solver exception/result with a profiling IO failure.
            try:
                if profiler is not None:
                    profiler.dump_stats(str(output / (ident + '.pstats')))
                data = recorder.export()
                temporary = output / (ident + '.json.tmp')
                temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
                temporary.replace(output / (ident + '.json'))
            except Exception as exc:
                import warnings
                warnings.warn('TAMP profile could not be saved: %s' % exc)
    return wrapped
