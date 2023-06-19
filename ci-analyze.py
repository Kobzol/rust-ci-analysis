import contextlib
import dataclasses
import datetime
import json
import multiprocessing
import os
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import List, Iterator, Optional, Iterable, Dict, Callable, Tuple, Union

import pandas as pd
import requests
import tqdm
import typer

CI_JOBS = (
    "aarch64-gnu",
    "arm-android",
    "armhf-gnu",
    "dist-aarch64-linux",
    "dist-android",
    "dist-arm-linux",
    "dist-armhf-linux",
    "dist-armv7-linux",
    "dist-i586-gnu-i586-i686-musl",
    "dist-i686-linux",
    "dist-loongarch64-linux",
    "dist-mips-linux",
    "dist-mips64-linux",
    "dist-mips64el-linux",
    "dist-mipsel-linux",
    "dist-powerpc-linux",
    "dist-powerpc64-linux",
    "dist-powerpc64le-linux",
    "dist-riscv64-linux",
    "dist-s390x-linux",
    "dist-various-1",
    "dist-various-2",
    "dist-x86_64-freebsd",
    "dist-x86_64-illumos",
    "dist-x86_64-linux",
    "dist-x86_64-linux-alt",
    "dist-x86_64-musl",
    "dist-x86_64-netbsd",
    "i686-gnu",
    "i686-gnu-nopt",
    "mingw-check",
    "test-various",
    "wasm32",
    "x86_64-gnu",
    "x86_64-gnu-stable",
    "x86_64-gnu-aux",
    "x86_64-gnu-debug",
    "x86_64-gnu-distcheck",
    "x86_64-gnu-llvm-16",
    "x86_64-gnu-llvm-15",
    "x86_64-gnu-llvm-14",
    "x86_64-gnu-llvm-14-stage1",
    "x86_64-gnu-nopt",
    "x86_64-gnu-tools",
    "dist-x86_64-apple",
    "dist-apple-various",
    "dist-x86_64-apple-alt",
    "x86_64-apple-1",
    "x86_64-apple-2",
    "dist-aarch64-apple",
    "x86_64-msvc",
    "x86_64-msvc-1",
    "x86_64-msvc-2",
    "i686-msvc",
    "i686-msvc-1",
    "i686-msvc-2",
    "x86_64-msvc-cargo",
    "x86_64-msvc-tools",
    "i686-mingw-1",
    "i686-mingw-2",
    "x86_64-mingw-1",
    "x86_64-mingw-2",
    "dist-x86_64-msvc",
    "dist-i686-msvc",
    "dist-aarch64-msvc",
    "dist-i686-mingw",
    "dist-x86_64-mingw",
    "dist-x86_64-msvc-alt"
)

CURRENT_DIR = Path(__file__).absolute().parent
CACHE_DIR = CURRENT_DIR / ".cache"
PATH_TO_RUSTC = CURRENT_DIR.parent

app = typer.Typer()

Job = str


@dataclasses.dataclass(frozen=True)
class Test:
    name: str
    outcome: str


@dataclasses.dataclass(frozen=True)
class TestSuite:
    tests: List[Test]


class BuildStep:
    def __init__(self, type: str, children: List["BuildStep"], duration: float, duration_excluding_children: float,
                 stage: Optional[int] = None, tests: Optional[List[Test]] = None):
        self.type = type
        self.children = children
        self.duration = duration
        self.duration_excluding_children = duration_excluding_children
        self.stage = stage
        self.tests = tests

    def find_all_by_filter(self, filter: Callable[["BuildStep"], bool]) -> Iterator["BuildStep"]:
        if filter(self):
            yield self
            return
        for child in self.children:
            yield from child.find_all_by_filter(filter)

    def duration_by_filter(self, filter: Callable[["BuildStep"], bool]) -> float:
        children = tuple(self.find_all_by_filter(filter))
        return sum(step.duration for step in children)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def iterate_all_children(self) -> Iterable["BuildStep"]:
        for child in self.children:
            yield child
            yield from child.iterate_all_children()

    def iterate_all_tests(self) -> Iterable[Test]:
        if self.tests is not None:
            for test in self.tests:
                yield test
        for child in self.children:
            yield from child.iterate_all_tests()

    def __repr__(self):
        return f"BuildStep(type={self.type}, duration={self.duration}, children={len(self.children)})"


STAGE_REGEX = re.compile(r".*stage: (\d).*")


def load_metrics(metrics, parse_tests: bool = False) -> List[BuildStep]:
    def parse_invocation(invocation) -> BuildStep:
        def parse(entry) -> Optional[Union[BuildStep, TestSuite]]:
            def normalize_test_name(name: str) -> str:
                return name.replace("\\", "/")

            if "kind" not in entry:
                return None
            elif parse_tests and entry["kind"] == "test_suite":
                tests = entry["tests"]
                tests = [Test(name=normalize_test_name(t["name"]), outcome=t["outcome"]) for t in tests]
                return TestSuite(tests=tests)
            elif parse_tests and entry["kind"] == "test":
                tests = [Test(name=normalize_test_name(entry["name"]), outcome=entry["outcome"])]
                return TestSuite(tests=tests)
            elif entry["kind"] == "rustbuild_step":
                type = entry.get("type", "")
                duration_excluding_children = entry.get("duration_excluding_children_sec", 0)
                duration = duration_excluding_children
                children = []
                tests = []

                stage = STAGE_REGEX.match(entry.get("debug_repr", ""))
                if stage is not None:
                    stage = int(stage.group(1))

                for child in entry.get("children", ()):
                    step = parse(child)
                    if step is not None:
                        if isinstance(step, TestSuite):
                            tests.extend(step.tests)
                        elif isinstance(step, BuildStep):
                            children.append(step)
                            duration += step.duration
                        else:
                            assert False
                return BuildStep(type=type, children=children, duration=duration,
                                 duration_excluding_children=duration_excluding_children, stage=stage,
                                 tests=tests)
            return None

        children = [parse(child) for child in invocation.get("children", ())]
        total_duration = invocation.get("duration_including_children_sec", 0)
        return BuildStep(
            type="root",
            children=children,
            duration=total_duration,
            duration_excluding_children=total_duration - sum(c.duration for c in children)
        )

    return [parse_invocation(invocation) for invocation in metrics["invocations"]]


def download_metrics(sha: str, job: str):
    url = f"https://ci-artifacts.rust-lang.org/rustc-builds/{sha}/metrics-{job}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_metrics(sha: str, job: str, parse_tests: bool) -> List[BuildStep]:
    cache_dir = CACHE_DIR / sha
    cache_dir.mkdir(parents=True, exist_ok=True)
    metric_path = cache_dir / f"{job}.json"
    if metric_path.is_file():
        # Missing metrics
        if os.stat(metric_path).st_size == 0:
            return []
        with open(metric_path) as f:
            data = json.load(f)
    else:
        try:
            data = download_metrics(sha, job)
        except BaseException as e:
            with open(metric_path, "w"):
                # Create empty file
                pass
            print(e)
            return []
        with open(metric_path, "w") as f:
            json.dump(data, f)
    return load_metrics(data, parse_tests=parse_tests)


class MetricDownloader:
    def __init__(self, pool: multiprocessing.Pool):
        self.pool = pool

    def get_metrics_for_sha(self, sha: str, jobs: Iterable[str], parse_tests: bool) -> Dict[str, BuildStep]:
        result = {}
        for (job, metric) in zip(jobs, self.pool.starmap(get_metrics, list((sha, job, parse_tests) for job in jobs))):
            if metric is not None:
                result[job] = metric
        return result


@contextlib.contextmanager
def create_downloader() -> Iterable[MetricDownloader]:
    with multiprocessing.Pool() as pool:
        yield MetricDownloader(pool)


@dataclasses.dataclass
class Commit:
    sha: str
    date: datetime.datetime


def get_commits_from_last_n_days(days: int) -> List[Commit]:
    """
    Return rust-lang/rust merge commits for the last `days` days.
    """
    from git import Repo

    def iterate_merge_commits():
        repo = Repo(PATH_TO_RUSTC)
        commit = repo.heads.master.commit
        while True:
            if commit.message.startswith("Auto merge"):
                yield commit
            for parent in commit.parents:
                if parent.message.startswith("Auto merge") and commit.binsha != parent.binsha:
                    commit = parent
                    break

    commit_iter = iterate_merge_commits()
    commits = [next(commit_iter)]
    while len(commits) < days:
        commit = next(commit_iter)
        commit_date = commit.committed_datetime
        previous_date = commits[-1].committed_datetime
        if commit_date.date() != previous_date.date():
            commits.append(commit)
    return [Commit(sha=commit.hexsha, date=commit.committed_datetime) for commit in commits]


def calculate_test_duration(step: BuildStep) -> (float, float):
    """
    Returns (test run duration, test build duration)
    """
    run_duration = step.duration
    build_duration = 0

    def iterate(item):
        nonlocal run_duration, build_duration

        for child in item.children:
            if "ToolBuild" in child.type or "TestHelpers" in child.type:
                run_duration -= child.duration
                build_duration += child.duration
            elif child.type in ("bootstrap::compile::Rustc", "bootstrap::compile::Assemble"):
                run_duration -= child.duration
            else:
                iterate(child)

    iterate(step)

    return (run_duration, build_duration)


@dataclasses.dataclass
class InvocationResult:
    llvm: float
    rustc_stage_1: float
    rustc_stage_2: float
    test_run: float
    test_build: float
    suites: Dict[str, float]
    total: float


def aggregate_step(metrics: BuildStep) -> InvocationResult:
    llvm = metrics.duration_by_filter(lambda step: step.type == "bootstrap::llvm::Llvm")
    rustc_stage_1 = metrics.duration_by_filter(
        lambda step: step.type == "bootstrap::compile::Rustc" and step.stage == 0)
    rustc_stage_2 = metrics.duration_by_filter(
        lambda step: step.type == "bootstrap::compile::Rustc" and step.stage == 1)
    test_steps = list(metrics.find_all_by_filter(lambda step: step.type.startswith("bootstrap::test::")))
    test_durations = [calculate_test_duration(step) for step in test_steps]

    test_run = sum(t[0] for t in test_durations)
    test_build = sum(t[1] for t in test_durations)
    test_total = sum(s.duration for s in test_steps)

    assert test_run + test_build <= test_total + 10

    suites = {}
    for test_step in test_steps:
        suite_name = test_step.type[len("bootstrap::test::"):]
        if test_step.duration > 10:
            suites[suite_name] = calculate_test_duration(test_step)[0]

    return InvocationResult(
        llvm=llvm,
        rustc_stage_1=rustc_stage_1,
        rustc_stage_2=rustc_stage_2,
        test_run=test_run,
        test_build=test_build,
        suites=suites,
        total=metrics.duration
    )


def print_step_table(step: BuildStep):
    substeps: List[Tuple[int, BuildStep]] = []

    def visit(step: BuildStep, level: int):
        substeps.append((level, step))
        for child in step.children:
            visit(child, level=level + 1)

    visit(step, 0)

    output = StringIO()
    for (level, step) in substeps:
        label = f"{'.' * level}{step.type}"
        print(f"{label:<65}{step.duration:>8.2f}s", file=output)
    print(f"Build step durations\n{output.getvalue()}")


@app.command()
def download_ci_durations(days: int = 30, output: Path = "result.csv"):
    """
    Downloads the metrics.json files from the last `days` of master merge commits.
    Analyzes the metrics and stores durations of interesting bootstrap steps into `output`.
    """
    commits = get_commits_from_last_n_days(days)
    items: List[Tuple[str, Commit, List[InvocationResult]]] = []
    with create_downloader() as downloader:
        for commit in tqdm.tqdm(commits):
            response = downloader.get_metrics_for_sha(commit.sha, CI_JOBS, parse_tests=False)
            for (job, metrics) in response.items():
                metrics: List[BuildStep] = metrics
                if len(metrics) > 0:
                    results = [aggregate_step(step) for step in metrics]
                    items.append((job, commit, results))

    known_suites = set()
    for (_, _, results) in items:
        for result in results:
            known_suites |= set(result.suites.keys())

    data = defaultdict(list)
    for (job, commit, results) in items:
        data["timestamp"].append(int(commit.date.timestamp()))
        data["job"].append(job)
        data["llvm"].append(sum(r.llvm for r in results))
        data["rustc-1"].append(sum(r.rustc_stage_1 for r in results))
        data["rustc-2"].append(sum(r.rustc_stage_2 for r in results))
        data["test-run"].append(sum(r.test_run for r in results))
        data["test-build"].append(sum(r.test_build for r in results))
        data["total"].append(sum(r.total for r in results))

        # We need to make sure that all suites are added for each job, otherwise creation of the DataFarme below will
        # fail.
        added_suites = set()
        for (suite, duration) in results[-1].suites.items():
            data[f"suite-{suite}"].append(duration)
            added_suites.add(suite)
        for suite in known_suites:
            if suite not in added_suites:
                data[f"suite-{suite}"].append(0)

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)


def get_tests(commit: str) -> List[Tuple[Job, Test]]:
    tests = []
    with create_downloader() as downloader:
        response = downloader.get_metrics_for_sha(commit, CI_JOBS, parse_tests=True)
        for (job, metrics) in response.items():
            metrics: List[BuildStep] = metrics
            for step in metrics:
                for test in step.iterate_all_tests():
                    tests.append((job, test))
    return tests


@app.command()
def analyze_tests(commit: Optional[str] = None):
    """
    Analyzes tests for the given `commit`.
    Prints results to output.
    """
    if commit is None:
        commit = get_commits_from_last_n_days(1)[0]
    tests = get_tests(commit.sha)

    # Test to count
    test_to_count = defaultdict(int)
    for (job, test) in tests:
        if test.outcome == "passed":
            test_to_count[test.name] += 1
    items = list(test_to_count.items())
    most_tested = sorted(items, key=lambda item: item[1], reverse=True)

    import seaborn as sns
    import matplotlib.pyplot as plt
    values = [v[1] for v in most_tested]
    sns.histplot(values)
    plt.savefig("test-histogram.png")

    print("5 most and least executed tests")
    for (test, count) in most_tested[:5]:
        print(f"{test}: {count}x")
    for (test, count) in most_tested[-5:]:
        print(f"{test}: {count}x")
    print()

    # Job to outcome count
    print("Job to test count")
    job_to_outcome = defaultdict(lambda: defaultdict(int))
    for (job, test) in tests:
        job_to_outcome[job][test.outcome] += 1
    for (job, outcomes) in sorted(job_to_outcome.items(), key=lambda item: item[0]):
        passed = outcomes.get("passed", 0)
        ignored = outcomes.get("ignored", 0)
        print(f"{job}: passed={passed}, ignored={ignored}")


if __name__ == "__main__":
    app()
