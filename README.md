# HLP time and resource analysis

Code and archived inputs for analyzing shuttle-bus capacity, protected scheduling, and requested physical resources in hierarchical quantum error correction. No manuscript, presentation, or publication figures are included.

## Quick verification

From the repository root, with Python 3.10 or later:

```sh
python scripts/verify.py
python scripts/run_tests.py
```

The first command uses the standard library. It checks file integrity, independently recomputes 40,960 design/concurrency combinations, validates 15 schedules against the saved physical endpoints, and exports exact timing choices to `output/`. The second command requires NumPy, Stim, and pytest. It checks geometry, physical resources, fault-signature auditing, and rational timing boundaries. Neither command performs noisy quantum sampling or decoding.

The studied family has n=16, inner distance 5, uniform port capacities 4 through 16, bus counts 1 through 16, a 18,928-qubit budget, and concurrency caps 25 through 6,400 in increments of 25. Gates cannot move across the retained protection or group boundaries. Each occupied port is an atomic bundle of 25 disjoint physical CX gates. The scheduling optimum is restricted to this family.

For 16 buses, ports 4/6/8 give protected-round counts 108/88/68 and qubit counts 15,728/17,328/18,928. At concurrency 1,600 their cross-block gate depths are 16/20/16. These counts are not calibrated hardware latency or equal-reliability speedups. Timing uses `T/t_SE = R + rho*L` with equal nominal boundary-stage durations; exact rational envelopes preserve ties.

## Files

- `src/hlp_artifact/`: exact finite-domain timing analysis and independent archive checker.
- `src/experiments/`: capacity, physical schedule, resource, and circuit routines.
- `results/`: selected original scientific inputs and circuit evidence. Large text files use lossless gzip compression.
- `vendor/ShuttlebusMemory/`: required upstream circuit modules and Apache-2.0 license.
- `provenance/`: source mapping, original and distributed hashes, upstream revision, and packaging transformations.
- `scripts/`: integrity checks, test runner, and decompression utility.

## Optional circuit replay

Historical tagged full-circuit results used **Stim 1.15.0**, noise probability 0.001, and seed 41819. Per-file metadata records historical interpreter versions, which differ between runs. Packaging checks use Python 3.10.18 and NumPy 1.26.4. The supplied scripts never install or change packages. For tagged circuit replay use an existing environment with Stim 1.15.0; Stim 1.14.0 can run the structural tests but cannot reconstruct tagged circuits.

```sh
python scripts/unpack.py
# Set PYTHONPATH=src using the syntax of your shell.
python -m experiments.capacity_validation.circuit --n 8 --ports 8 --rounds 2
python -m experiments.concurrency_validation.rebuild --n 8 --ports 8 --cap 800
```

`unpack.py` checks compressed input integrity before unpacking. Circuit regeneration is optional and can consume substantial memory and time. It writes result files; use a separate checkout if preserving the distributed archive unchanged. The two n8 capacity circuits and four ideal serialized circuits are already supplied. The n16 four-port source is supplied for serialization; no full n16 six/eight-port circuit validation is claimed.

The n8 archives contain deterministic-detector/noiseless checks and undecomposed detector-error-model signature checks. Signature checks detect undetected single terms and matching detector signatures with different logical parities; they do not prove distance five, decoder performance, or arbitrary two-fault correction. Serialized circuits omit noise because added idle-slot noise is uncalibrated. No threshold, logical failure-rate improvement, hardware routing feasibility, or independent control-channel count is established.

## Provenance and licenses

Upstream: Zihan-Chen-PhMA/ShuttlebusMemory, revision recorded in `provenance/upstream.json`. Included upstream modules retain Apache-2.0 licensing. Original analysis code is MIT-licensed; archived numerical records are made available under the repository MIT license, subject to upstream rights for generated circuit representations. See `NOTICE`.

Machine-specific historical paths in JSON metadata have been normalized; numerical fields, seeds, and original source digests are preserved. `provenance/assembly.json` records original and packaged hashes and identifies these transformations. Historical source digest entries describe the original experiment, not the portable source revision. The release manifest checks the actual distributed files.
