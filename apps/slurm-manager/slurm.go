package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type JobState string

const (
	JobPending     JobState = "pending"
	JobRunning     JobState = "running"
	JobUnavailable JobState = "unavailable"
	JobTerminal    JobState = "terminal"
	JobPreempted   JobState = "preempted"
)

type ManagedJob struct {
	ID            string        `json:"id"`
	ResourceClass ResourceClass `json:"resource_class"`
	State         JobState      `json:"state"`
	SlurmState    string        `json:"slurm_state"`
	SlurmReason   string        `json:"slurm_reason,omitempty"`
}

type SlurmClient interface {
	ListManagedJobs(context.Context) ([]ManagedJob, error)
	Submit(context.Context, ResourceClass, ResourceConfig) (string, error)
}

type CommandRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

type ExecCommandRunner struct{}

func (ExecCommandRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	command := exec.CommandContext(ctx, name, args...)
	command.Env = environmentWithoutAllocationToken(os.Environ())
	output, err := command.CombinedOutput()
	if err != nil {
		detail := strings.TrimSpace(string(output))
		if len(detail) > 4096 {
			detail = detail[:4096] + "..."
		}
		if detail != "" {
			return nil, fmt.Errorf("%s failed: %w: %s", name, err, detail)
		}
		return nil, fmt.Errorf("%s failed: %w", name, err)
	}
	return output, nil
}

func environmentWithoutAllocationToken(environment []string) []string {
	prefix := allocationTokenEnv + "="
	filtered := make([]string, 0, len(environment))
	for _, value := range environment {
		if !strings.HasPrefix(value, prefix) {
			filtered = append(filtered, value)
		}
	}
	return filtered
}

type CommandSlurmClient struct {
	config SlurmConfig
	runner CommandRunner
	logger *slog.Logger
	now    func() time.Time
}

func newCommandSlurmClient(config SlurmConfig, runner CommandRunner, logger *slog.Logger) *CommandSlurmClient {
	return &CommandSlurmClient{config: config, runner: runner, logger: logger, now: time.Now}
}

// ListManagedJobs lists Slurm jobs owned by this manager.
//
// Responsibilities:
// - Read active jobs from squeue and recent status evidence from sacct.
// - Return only jobs with exact owner-tag and resource-class comments.
// - Preserve recent terminal and preempted accounting records for status output.
//
// Rules:
// - Treat squeue failures and unknown owned Slurm states as errors.
// - Treat sacct failures as errors only when accounting is required by config.
// - Keep squeue as the authoritative active-capacity source.
func (c *CommandSlurmClient) ListManagedJobs(ctx context.Context) ([]ManagedJob, error) {
	commandContext, cancel := context.WithTimeout(ctx, c.config.CommandTimeout.Duration)
	defer cancel()
	squeueOutput, err := c.runner.Run(
		commandContext,
		c.config.SqueueBin,
		"--noheader",
		"--user", c.config.User,
		"--format=%i|%T|%r|%k",
	)
	if err != nil {
		return nil, fmt.Errorf("list active Slurm jobs: %w", err)
	}
	active, err := parseJobRows(string(squeueOutput), c.config.OwnerTag, true)
	if err != nil {
		return nil, fmt.Errorf("parse squeue output: %w", err)
	}

	commandContext, cancel = context.WithTimeout(ctx, c.config.CommandTimeout.Duration)
	defer cancel()
	start := c.now().Add(-c.config.AccountingLookback.Duration).Format("2006-01-02T15:04:05")
	sacctOutput, err := c.runner.Run(
		commandContext,
		c.config.SacctBin,
		"--noheader",
		"--parsable2",
		"--starttime", start,
		"--user", c.config.User,
		"--format=JobIDRaw,State,Reason,Comment",
	)
	if err != nil {
		if c.config.AccountingRequired {
			return nil, fmt.Errorf("list recent Slurm accounting jobs: %w", err)
		}
		return active, nil
	}
	historical, err := parseJobRows(string(sacctOutput), c.config.OwnerTag, false)
	if err != nil {
		return nil, fmt.Errorf("parse sacct output: %w", err)
	}

	jobs := make([]ManagedJob, 0, len(active)+len(historical))
	jobs = append(jobs, active...)
	activeIDs := make(map[string]struct{}, len(active))
	for _, job := range active {
		activeIDs[job.ID] = struct{}{}
	}
	for _, job := range historical {
		if _, exists := activeIDs[job.ID]; !exists {
			jobs = append(jobs, job)
		}
	}
	return jobs, nil
}

func (c *CommandSlurmClient) Submit(ctx context.Context, class ResourceClass, resource ResourceConfig) (string, error) {
	comment := c.config.OwnerTag + ":" + string(class)
	exportFile, err := writeSubmissionExportFile(c.config, resource)
	if err != nil {
		return "", fmt.Errorf("prepare %s allocation environment: %w", class, err)
	}
	defer os.Remove(exportFile)
	args := []string{
		"--parsable",
		"--comment=" + comment,
		"--partition=" + resource.Partition,
		"--time=" + resource.Walltime,
		"--export-file=" + exportFile,
	}
	if c.config.Account != "" {
		args = append(args, "--account="+c.config.Account)
	}
	if resource.QOS != "" {
		args = append(args, "--qos="+resource.QOS)
	}
	args = append(args, resource.Template)

	commandContext, cancel := context.WithTimeout(ctx, c.config.CommandTimeout.Duration)
	defer cancel()
	output, err := c.runner.Run(commandContext, c.config.SbatchBin, args...)
	if err != nil {
		return "", fmt.Errorf("submit %s allocation: %w", class, err)
	}
	jobID := strings.TrimSpace(strings.SplitN(string(output), ";", 2)[0])
	if _, err := strconv.ParseUint(jobID, 10, 64); err != nil {
		return "", fmt.Errorf("submit %s allocation: unexpected sbatch job ID %q", class, jobID)
	}
	return jobID, nil
}

func writeSubmissionExportFile(config SlurmConfig, resource ResourceConfig) (string, error) {
	directory := filepath.Dir(config.LockFile)
	file, err := os.CreateTemp(directory, ".slurm-manager-export-*")
	if err != nil {
		return "", err
	}
	path := file.Name()
	cleanup := func() {
		_ = file.Close()
		_ = os.Remove(path)
	}
	if err := file.Chmod(0o600); err != nil {
		cleanup()
		return "", err
	}
	values := []string{
		"CONCORD_WORKER_COMMON_ENV_FILE=" + config.CommonEnvFile,
		"CONCORD_WORKER_ENV_FILE=" + resource.WorkerEnvFile,
		"CONCORD_WORKER_IDLE_TIMEOUT_SECONDS=" + strconv.FormatInt(int64(resource.IdleTimeout.Duration/time.Second), 10),
	}
	if _, err := file.Write([]byte(strings.Join(values, "\x00") + "\x00")); err != nil {
		cleanup()
		return "", err
	}
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return path, nil
}

func parseJobRows(output, ownerTag string, activeSource bool) ([]ManagedJob, error) {
	var jobs []ManagedJob
	seen := make(map[string]struct{})
	for lineNumber, raw := range strings.Split(output, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		fields := strings.SplitN(line, "|", 4)
		if len(fields) != 4 {
			return nil, fmt.Errorf("line %d has %d fields, expected 4", lineNumber+1, len(fields))
		}
		jobID := strings.TrimSpace(fields[0])
		if strings.Contains(jobID, ".") {
			continue
		}
		reason := strings.TrimSpace(fields[2])
		class, managed := parseOwnership(strings.TrimSpace(fields[3]), ownerTag)
		if !managed {
			continue
		}
		state, normalized, err := normalizeSlurmState(fields[1], reason)
		if err != nil {
			return nil, fmt.Errorf("managed job %s: %w", jobID, err)
		}
		if !activeSource && (state == JobPending || state == JobRunning) {
			continue
		}
		if _, duplicate := seen[jobID]; duplicate {
			continue
		}
		seen[jobID] = struct{}{}
		jobs = append(jobs, ManagedJob{
			ID:            jobID,
			ResourceClass: class,
			State:         state,
			SlurmState:    normalized,
			SlurmReason:   reason,
		})
	}
	return jobs, nil
}

func parseOwnership(comment, ownerTag string) (ResourceClass, bool) {
	prefix := ownerTag + ":"
	if !strings.HasPrefix(comment, prefix) {
		return "", false
	}
	class := ResourceClass(strings.TrimPrefix(comment, prefix))
	return class, isManagedResourceClass(class)
}

func normalizeSlurmState(raw, reason string) (JobState, string, error) {
	normalized := strings.ToUpper(strings.TrimSpace(raw))
	if field, _, found := strings.Cut(normalized, " "); found {
		normalized = field
	}
	normalized = strings.TrimSuffix(normalized, "+")
	if normalized == "PENDING" && isHeldReason(reason) {
		return JobUnavailable, normalized, nil
	}
	switch normalized {
	case "PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_FED":
		return JobPending, normalized, nil
	case "RUNNING", "RESIZING":
		return JobRunning, normalized, nil
	case "COMPLETING", "REQUEUE_HOLD", "RESV_DEL_HOLD", "SIGNALING", "STOPPED", "SUSPENDED":
		return JobUnavailable, normalized, nil
	case "PREEMPTED":
		return JobPreempted, normalized, nil
	case "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "REVOKED", "SPECIAL_EXIT", "STAGE_OUT", "TIMEOUT":
		return JobTerminal, normalized, nil
	default:
		return "", normalized, errors.New("unknown Slurm state " + strconv.Quote(normalized))
	}
}

func isHeldReason(reason string) bool {
	normalized := strings.ToLower(strings.TrimSpace(reason))
	return strings.Contains(normalized, "jobheld") || strings.HasSuffix(normalized, "hold") || strings.Contains(normalized, "held")
}
