import contextlib
import json
import multiprocessing
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Iterator, Optional, Iterable, Dict, Callable

import pandas as pd
import requests
import tqdm

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
    "x86_64-msvc-1",
    "x86_64-msvc-2",
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


class BuildStep:
    def __init__(self, type: str, children: List["BuildStep"], duration: float, stage: Optional[int] = None):
        self.type = type
        self.children = children
        self.duration = duration
        self.stage = stage

    def find_all_by_filter(self, filter: Callable[["BuildStep"], bool]) -> Iterator["BuildStep"]:
        if filter(self):
            yield self
            return
        for child in self.children:
            yield from child.find_all_by_filter(filter)

    def duration_by_filter(self, filter: Callable[["BuildStep"], bool]) -> float:
        children = tuple(self.find_all_by_filter(filter))
        return sum(step.duration for step in children)

    def __repr__(self):
        return f"BuildStep(type={self.type}, duration={self.duration}, children={len(self.children)})"


STAGE_REGEX = re.compile(r".*stage: (\d).*")


def load_metrics(metrics) -> BuildStep:
    assert len(metrics["invocations"])
    invocation = metrics["invocations"][-1]

    def parse(entry) -> Optional[BuildStep]:
        if "kind" not in entry or entry["kind"] != "rustbuild_step":
            return None
        type = entry.get("type", "")
        duration = entry.get("duration_excluding_children_sec", 0)
        children = []

        stage = STAGE_REGEX.match(entry.get("debug_repr", ""))
        if stage is not None:
            stage = int(stage.group(1))

        for child in entry.get("children", ()):
            step = parse(child)
            if step is not None:
                children.append(step)
                duration += step.duration
        return BuildStep(type=type, children=children, duration=duration, stage=stage)

    children = [parse(child) for child in invocation.get("children", ())]
    return BuildStep(
        type="root",
        children=children,
        duration=invocation.get("duration_including_children_sec", 0)
    )


def download_metrics(sha: str, job: str):
    url = f"https://ci-artifacts.rust-lang.org/rustc-builds/{sha}/metrics-{job}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_metrics(sha: str, job: str) -> Optional[BuildStep]:
    cache_dir = CACHE_DIR / sha
    cache_dir.mkdir(parents=True, exist_ok=True)
    metric_path = cache_dir / f"{job}.json"
    if metric_path.is_file():
        # Missing metrics
        if os.stat(metric_path).st_size == 0:
            return None
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
            return None
        with open(metric_path, "w") as f:
            json.dump(data, f)
    return load_metrics(data)


class MetricDownloader:
    def __init__(self, pool: multiprocessing.Pool):
        self.pool = pool

    def get_metrics_for_sha(self, sha: str, jobs: Iterable[str]) -> Dict[str, BuildStep]:
        result = {}
        for (job, metric) in zip(jobs, self.pool.starmap(get_metrics, list((sha, job) for job in jobs))):
            if metric is not None:
                result[job] = metric
        return result


@contextlib.contextmanager
def create_downloader() -> Iterable[MetricDownloader]:
    with multiprocessing.Pool() as pool:
        yield MetricDownloader(pool)


def get_shas_from_last_n_days(days: int) -> List[str]:
    """
    Return rust-lang/rust merge commit SHAs for the last `days` days.
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
    return [commit.hexsha for commit in commits]


if __name__ == "__main__":
    data = defaultdict(list)
    shas = get_shas_from_last_n_days(30)

    with create_downloader() as downloader:
        for commit in tqdm.tqdm(shas):
            response = downloader.get_metrics_for_sha(commit, CI_JOBS)
            for (job, metrics) in response.items():
                metrics: BuildStep = metrics
                llvm = metrics.duration_by_filter(lambda step: step.type == "bootstrap::llvm::Llvm")
                rustc_stage_1 = metrics.duration_by_filter(lambda step: step.type == "bootstrap::compile::Rustc" and step.stage == 0)
                rustc_stage_2 = metrics.duration_by_filter(lambda step: step.type == "bootstrap::compile::Rustc" and step.stage == 1)
                tests = metrics.duration_by_filter(lambda step: step.type.startswith("bootstrap::test::"))
                total = metrics.duration
                data["job"].append(job)
                data["llvm"].append(llvm)
                data["rustc-1"].append(rustc_stage_1)
                data["rustc-2"].append(rustc_stage_2)
                data["tests"].append(tests)
                data["total"].append(total)
    df = pd.DataFrame(data)
    df.to_csv("result.csv", index=False)
