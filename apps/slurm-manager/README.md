# Slurm allocation manager

**TODO: De-slopify this and update for new approach to Slurm partition selection.**

`apps/slurm-manager` is a one-shot Go reconciler for login-node cron. It polls
Concord's authenticated worker-demand endpoint and supplies Slurm allocations
for `cpu`, `cpu-preempt`, `gpu`, and `gpu-preempt`. It never polls Hatchet,
schedules workflow actions, cancels healthy allocations because demand fell, or
receives Hatchet/worker/database credentials.

## Build and configure

Go 1.24 or newer:

```bash
cd apps/slurm-manager
go test ./...
go build -o /opt/concord/bin/concord-slurm-manager .
```

Copy [`../../deploy/slurm-manager/config.json.example`](../../deploy/slurm-manager/config.json.example)
to an allocation-user-owned `0600` regular file and adapt it to the site.
Startup rejects symlink/special or group/world-writable config/templates and
requires worker environment files to be regular `0600` files. Keep
`CONCORD_ALLOCATION_MANAGER_TOKEN` in a separate protected environment file.
The token must be distinct from worker and browser credentials.

The configuration owns:

- Concord URL, request timeout, accepted snapshot age, and clock skew;
- `squeue`, `sacct`, and `sbatch` commands, Slurm user/account, command timeout,
  accounting lookback/required policy, shared process lock, and stable ownership tag;
- protected common/resource worker environment-file paths; and
- per-class minimum/maximum allocations, worker idle-timeout contract,
  walltime, template, partition, and QoS.

Each allocation starts one Concord worker process, and each process owns one
Hatchet slot. Allocations mount cluster scratch and protected science assets, not
API application storage. `idle_timeout` records the worker-side drain policy: healthy excess
capacity drains through worker idle timeout or walltime. A zero minimum allows all
excess jobs to leave; a positive minimum is replenished. Keep idle timeout
shorter than walltime at deployment.

The manager submits the existing protected worker templates with a transient
`0600`, NUL-delimited `--export-file` containing only protected file paths and
the idle timeout; the file is removed after `sbatch` returns. Templates omit an
export directive because it would override `--export-file`; the wrapper also
scrubs inherited variables before loading protected files. Token values remain
inside protected files. Managed jobs have exact Slurm comments
`<owner_tag>:<resource-class>`. Jobs without one of those four exact comments
are ignored.

## Reconciliation

Reconcile-local behavior is documented in [`reconcile.go`](reconcile.go). That
source owns the desired allocation calculation, active-capacity counting,
per-class Slurm relisting, deficit submission, and fail-closed behavior.

Sites without Slurm accounting should keep `accounting_required=false`; `squeue`
remains the authoritative active-capacity path. Set it true only where missing
historical evidence must fail the pass.

Use dry-run or machine-readable status before enabling cron:

```bash
set -a; . "$HOME/.config/concord/slurm-manager-token.env"; set +a
/opt/concord/bin/concord-slurm-manager \
  --config "$HOME/.config/concord/slurm-manager.json" --dry-run
/opt/concord/bin/concord-slurm-manager \
  --config "$HOME/.config/concord/slurm-manager.json" --status
```

Example five-minute cron entry:

```cron
*/5 * * * * set -a; . "$HOME/.config/concord/slurm-manager-token.env"; set +a; /opt/concord/bin/concord-slurm-manager --config "$HOME/.config/concord/slurm-manager.json" >> "$HOME/.local/state/concord/slurm-manager.log" 2>&1
```

The non-blocking file lock makes overlapping cron entries exit successfully
without polling or submitting. Put the lock on storage shared by every host that
may run this cron; one host-local lock cannot coordinate multiple login nodes.
Alert on non-zero exits. Logs are JSON and redact/omit credentials.

## Recovery

- Concord HTTP 401: rotate/fix only the allocation-manager token.
- Concord HTTP 503/stale response: fix Hatchet/API visibility; do not treat it as
  zero demand.
- Unknown Slurm state: preserve `squeue`/`sacct` output and add an explicit state
  mapping before resuming.
- Ambiguous `sbatch` failure: inspect exact ownership comments before retrying;
  the next normal pass discovers successful submissions whose response was lost.
- Preemption/timeout/node failure: no manual state repair; the next successful
  pass replenishes the deficit.
- Duplicate jobs: stop extra cron hosts, verify they share one lock path and
  ownership tag, then allow excess healthy jobs to drain rather than cancelling
  active work.
