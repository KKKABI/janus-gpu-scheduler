# Reproducible benchmark harness

Model construction, weights, inputs, warm-up counts and iteration counts follow the matching files under `/public_0/LYX/PriorityOpara_v0/examples`.

Exception: the reference `yolov8_example.py` captures only `model.model[0]`, which is one Conv block. The corrected `YOLOv8x` task captures the complete Ultralytics `DetectionModel` and uses the matching frozen `DetectionModel_...trace.json` profile. Old BackboneWrapper results remain historical proxy data and must not be labeled as full YOLOv8x.

Activate the frozen environment and enter the isolated worktree:

    source /usr/local/Anaconda3/etc/profile.d/conda.sh
    conda activate opara
    cd /public_0/LYX/janus_repro

Print a deterministic plan without writing files or using the GPU:

    python experiments/run_matrix.py --dry-run --models GoogLeNet --variants Baseline TD+Janus TD+DRT --repeats 2

Verify the interpreter and all frozen manifests without running inference:

    python experiments/run_matrix.py --preflight

Run unit tests:

    python -m unittest experiments/test_harness.py

The default official command expands the six primary models, six primary variants and five independent process repeats. `TD+Janus` keeps the original Janus name-balance scorer and changes only the simulator, providing the direct control needed to isolate DRT:

    python experiments/run_matrix.py

Every configuration runs in a fresh process. A new immutable directory is created at `experiments/results/<run_id>`; an existing directory is never reused. Each task records its invocation, stdout/stderr, correctness evidence, every timing sample, statistics, exact profile hash, runtime versions, GPU telemetry and any failure.

The harness refuses an official run when the worktree is dirty, the interpreter is not `opara`, a frozen manifest changed, the exact profile is absent, another compute process uses the GPU, or output correctness fails. This prevents `GraphCapturer` from silently generating a profile during measurement.
