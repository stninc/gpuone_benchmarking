# Storage Benchmarks

Standalone [fio](https://fio.readthedocs.io/) job files for measuring local storage performance on a single host. Two dimensions are covered:

- **IOPS** — small-block (4K) random I/O, which stresses the device's ability to service many independent operations per second.
- **Throughput** — large-block (4M) random I/O, which stresses sustained bandwidth.

Each dimension has a read and a write variant.

## Job files

| File | Pattern | Block size | Measures |
| --- | --- | --- | --- |
| `iops-randread.fio` | random read | 4K | read IOPS |
| `iops-randwrite.fio` | random write | 4K | write IOPS |
| `throughput-randread.fio` | random read | 4M | read bandwidth |
| `throughput-randwrite.fio` | random write | 4M | write bandwidth |

## Common parameters

All four jobs share the following `[global]` settings:

| Setting | Value | Purpose |
| --- | --- | --- |
| `ioengine` | `libaio` | Linux asynchronous I/O |
| `direct` | `1` | Bypass the page cache so results reflect the device, not RAM |
| `iodepth` | `16` | Queue depth per job |
| `numjobs` | `32` | Parallel worker threads |
| `size` | `32G` | Working-set size per job |
| `time_based` + `runtime` | `600` s | Fixed 10-minute run regardless of size |
| `group_reporting` | on | Aggregate stats across all jobs |

Total working set per run is `numjobs * size` = **1 TiB**; ensure the target filesystem has enough free space.

## Requirements

- Linux with `libaio` available (most distributions; install `libaio1` / `libaio-devel` if missing).
- `fio` installed (`apt install fio`, `dnf install fio`, etc.).
- Run from a directory on the device you want to measure — fio creates its test files in the current working directory.
- Root or appropriate permissions if the target is a raw block device.

## Running

From the directory on the storage target under test:

```bash
fio /path/to/iops-randread.fio
fio /path/to/iops-randwrite.fio
fio /path/to/throughput-randread.fio
fio /path/to/throughput-randwrite.fio
```

Run one job file at a time so tests do not contend for the same device.

## Notes

- The write tests are destructive to the test files fio creates, but do not touch other data on the filesystem.
- For raw block-device testing, add a `filename=/dev/<device>` line under `[global]` or `[fiotest]` — **this will overwrite the device**.
- Adjust `size`, `numjobs`, or `iodepth` to match the device class under test (e.g. lower `numjobs` for consumer SSDs, higher `iodepth` for enterprise NVMe).
