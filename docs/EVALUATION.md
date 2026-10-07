# Store Traffic Monitor — Evaluation guide

Start with the smallest path that exercises the project. Distinguish source inspection,
syntax/build checks, functional behavior and domain validation when recording a result.

## Guided reading and demonstration

1. **Configure a controlled camera.** Supply authorized RTSP endpoints and zone boundaries. Review
the connection diagnostics before capturing data.

2. **Inspect stream processing.** The camera manager coordinates frames while the monitoring
system processes and displays them. Network reconnect behavior belongs to a different layer than
recognition accuracy.

3. **Review recognition records.** Face matching can register visitors and save embeddings/CSV
records. Existing artifacts are not a labeled accuracy benchmark.

4. **Check analytics output.** Inspect dwell-time, zone and heat-map reports. Validate
camera-to-store coordinates and false matches before interpreting cross-camera behavior.

## Declared checks

These commands/checks describe the intended verification path. Their presence in this
guide does not claim that they passed. See the dated evidence below and the README for setup.

```text
python -m compileall -q camera_manager.py traffic_analytics.py store_monitoring_system.py
```

## Evidence levels

| Level | What it establishes | What it does not establish |
| --- | --- | --- |
| Source review | A path exists in tracked code | Successful runtime behavior |
| Syntax/build | Parser/compiler accepts that path | End-to-end correctness |
| Behavioral check | A specific input/output case passed | Generalization beyond cases |
| Domain evaluation | Performance on a stated target setting | Other users/data/environments |

## What to record

- Commit, environment, dependency versions and date.
- Input provenance and whether data is synthetic, public or privately supplied.
- Absolute pass/fail/skip counts; keep failed cases and their root causes.
- Whether external services, hardware or a production deployment were actually exercised.
- Expected output and an artifact showing the observation.

## Review scenarios

- **Separate stream and recognition failures:** A connected camera is not proof of valid face
matching.

- **Local artifact storage:** CSV and pickle outputs need explicit trust, access and retention
controls.

- **Diagnostic tests are labeled:** Live camera scripts are not isolated unit tests.

## Documentation inspection — 7 October 2026

The documentation was traced to committed source and checked for local links, balanced
code fences and supported implementation claims. Historical notebook outputs remain labeled
as historical. Live provider access, private databases and hardware behavior are not inferred
from configuration or dependency files. Any fresh run is recorded separately in the README.

## Next evidence to collect

- Evaluate controlled footage with consenting participants.
- Measure false matches and coordinate calibration.
- Establish retention, access and deployment controls.
