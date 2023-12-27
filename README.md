# Rust CI analysis
Simple analysis scripts for [rust-lang/rust](https://github.com/rust-lang/rust) CI usage.

## Rust crate
The crate contains code for downloading workflow information from the GitHub API. It can download information about
completed `auto/try/PR` workflows from `rust-lang/rust` and `rust-lang-ci/rust` and show estimated durations and costs
(in USD) for individual jobs of each workflow.

## Python scripts
These scripts load commit information from a git checkout of `rust-lang/rust` and then downloads workflow metrics for
these commits from S3. They can chart durations of individual jobs and also steps within a job (Rust build, LLVM build,
test etc.).

### Installation
```console
$ python3 -m pip install -r requirements.txt
```

### Usage
The `ci-analyze.py` script assumes that a `rust` git checkout will be located in a directory one level above it.
You can change the location by modifying the `PATH_TO_RUSTC` variable.

- Download build metrics for last 14 days, and plot the results
    ```console
    $ python3 ci-analyze.py download-ci-durations 14
    $ python3 ci-plot.py build-durations                  # Plot total job durations
    $ python3 ci-plot.py step-durations --mode bootstrap  # Plot individual bootstrap step durations
    ```
- Analyze tests
    ```console
    $ python3 ci-analyze.py analyze-tests <commit>
    ```
    If you do not pass `<commit>`, the latest `rust-lang/rust` master merge commit will be used.
- Analyze the duration of all steps or a specific step of all CI jobs for a period of time or for
  a specific commit
    ```console
    $ python3 ci-analyze.py analyze-duration [--days 10] [--step bootstrap::test::RustAnalyzer]
    ```
