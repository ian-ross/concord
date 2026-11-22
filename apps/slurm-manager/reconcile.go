package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
)

type ClassResult struct {
	ResourceClass      ResourceClass `json:"resource_class"`
	QueuedTasks        int64         `json:"queued_tasks"`
	RunningTasks       int64         `json:"running_tasks"`
	HatchetCapacity    int64         `json:"hatchet_capacity"`
	MinAllocations     int           `json:"min_allocations"`
	MaxAllocations     int           `json:"max_allocations"`
	IdleTimeout        Duration      `json:"idle_timeout"`
	DesiredAllocations int           `json:"desired_allocations"`
	PendingAllocations int           `json:"pending_allocations"`
	RunningAllocations int           `json:"running_allocations"`
	Deficit            int           `json:"deficit"`
	SubmittedJobIDs    []string      `json:"submitted_job_ids"`
	Error              string        `json:"error,omitempty"`
}

type ReconcileResult struct {
	GeneratedAt string        `json:"generated_at"`
	DryRun      bool          `json:"dry_run"`
	Classes     []ClassResult `json:"classes"`
	ManagedJobs []ManagedJob  `json:"managed_jobs"`
}

type Reconciler struct {
	config Config
	demand DemandSource
	slurm  SlurmClient
	logger *slog.Logger
}

// Run reconciles managed Slurm worker allocations once.
//
// Responsibilities:
//   - Fetch one Concord demand snapshot and list managed Slurm jobs before
//     planning changes.
//   - Re-list managed jobs before each non-dry-run resource class so each class
//     sees earlier submissions in the same pass.
//   - Submit only the positive deficit for each managed resource class.
//
// Rules:
//   - Treat DemandSource.Fetch errors as fail-closed before Slurm inspection;
//     HTTPDemandClient.Fetch owns snapshot freshness and completeness validation.
//   - Treat the initial managed-job listing error as fail-closed before submissions.
//   - Record a class error and continue with later classes when that class relist
//     or submission fails; return a non-nil pass error after all classes are attempted.
//   - In dry-run mode, report deficits from the initial job listing and do not
//     submit allocations.
//
// References:
// - apps/slurm-manager/README.md
func (r *Reconciler) Run(ctx context.Context, dryRun bool) (ReconcileResult, error) {
	snapshot, err := r.demand.Fetch(ctx)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("read Concord worker demand: %w", err)
	}
	demandByResource := snapshot.byResource()
	initialJobs, err := r.slurm.ListManagedJobs(ctx)
	if err != nil {
		return ReconcileResult{}, err
	}
	result := ReconcileResult{
		GeneratedAt: snapshot.GeneratedAt.UTC().Format("2006-01-02T15:04:05.999999999Z"),
		DryRun:      dryRun,
		Classes:     make([]ClassResult, 0, len(managedResourceClasses)),
		ManagedJobs: append([]ManagedJob(nil), initialJobs...),
	}
	var failures []error

	for _, class := range managedResourceClasses {
		resource := r.config.Resources[class]
		demand := demandByResource[class]
		desired := desiredAllocations(demand, resource)
		jobs := initialJobs
		if !dryRun {
			jobs, err = r.slurm.ListManagedJobs(ctx)
			if err != nil {
				classErr := fmt.Errorf("refresh managed jobs before %s submission: %w", class, err)
				failures = append(failures, classErr)
				result.Classes = append(result.Classes, ClassResult{
					ResourceClass:      class,
					QueuedTasks:        demand.Queued,
					RunningTasks:       demand.Running,
					HatchetCapacity:    demand.WorkerCapacity,
					MinAllocations:     resource.MinAllocations,
					MaxAllocations:     resource.MaxAllocations,
					IdleTimeout:        resource.IdleTimeout,
					DesiredAllocations: desired,
					Error:              classErr.Error(),
				})
				continue
			}
		}
		pending, running := activeAllocationCounts(jobs, class)
		deficit := max(0, desired-pending-running)
		classResult := ClassResult{
			ResourceClass:      class,
			QueuedTasks:        demand.Queued,
			RunningTasks:       demand.Running,
			HatchetCapacity:    demand.WorkerCapacity,
			MinAllocations:     resource.MinAllocations,
			MaxAllocations:     resource.MaxAllocations,
			IdleTimeout:        resource.IdleTimeout,
			DesiredAllocations: desired,
			PendingAllocations: pending,
			RunningAllocations: running,
			Deficit:            deficit,
			SubmittedJobIDs:    []string{},
		}
		r.logger.Info("reconciled worker allocation class",
			"resource_class", class,
			"queued_tasks", demand.Queued,
			"running_tasks", demand.Running,
			"hatchet_capacity", demand.WorkerCapacity,
			"pending_allocations", pending,
			"running_allocations", running,
			"desired_allocations", desired,
			"deficit", deficit,
			"dry_run", dryRun,
		)
		if !dryRun {
			for range deficit {
				jobID, submitErr := r.slurm.Submit(ctx, class, resource)
				if submitErr != nil {
					classErr := fmt.Errorf("submit %s deficit: %w", class, submitErr)
					classResult.Error = classErr.Error()
					failures = append(failures, classErr)
					r.logger.Error("worker allocation submission failed", "resource_class", class, "error", submitErr)
					break
				}
				classResult.SubmittedJobIDs = append(classResult.SubmittedJobIDs, jobID)
				r.logger.Info("submitted worker allocation", "resource_class", class, "job_id", jobID)
			}
		}
		result.Classes = append(result.Classes, classResult)
	}

	if len(failures) > 0 {
		return result, errors.Join(failures...)
	}
	return result, nil
}

// desiredAllocations computes target allocations for one resource class.
//
// Responsibilities:
// - Sum queued and running Hatchet task demand for the class.
// - Clamp task demand to the configured minimum and maximum allocation bounds.
//
// Rules:
//   - Saturate queued-plus-running overflow at math.MaxInt64 before applying
//     allocation bounds.
func desiredAllocations(demand ResourceDemand, config ResourceConfig) int {
	tasks := demand.Queued
	if demand.Queued > math.MaxInt64-demand.Running {
		tasks = math.MaxInt64
	} else {
		tasks += demand.Running
	}
	desired := tasks
	if desired < int64(config.MinAllocations) {
		desired = int64(config.MinAllocations)
	}
	if desired > int64(config.MaxAllocations) {
		desired = int64(config.MaxAllocations)
	}
	return int(desired)
}

// activeAllocationCounts counts healthy managed capacity for one resource class.
//
// Responsibilities:
// - Count pending and running managed Slurm jobs separately.
// - Ignore jobs for other resource classes.
//
// Rules:
//   - Count only pending and running jobs as active capacity; unavailable,
//     terminal, and preempted records remain status evidence only.
func activeAllocationCounts(jobs []ManagedJob, class ResourceClass) (pending, running int) {
	for _, job := range jobs {
		if job.ResourceClass != class {
			continue
		}
		switch job.State {
		case JobPending:
			pending++
		case JobRunning:
			running++
		}
	}
	return pending, running
}
