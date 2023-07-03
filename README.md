# Rust CI analysis
Simple analysis scripts for [rust-lang/rust](https://github.com/rust-lang/rust) CI usage.

## Installation
```console
$ python3 -m pip install -r requirements.txt
```

## Usage
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
- Analyze the duration of a specific step for a period of time or for a specific commit
    ```console
    $ python3 ci-analyze.py analyze-step bootstrap::test::RustAnalyzer --days 10
    ```
