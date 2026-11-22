package main

import (
	"context"
	"io"
	"log/slog"
	"strconv"
	"time"
)

func validTestConfig() Config {
	resources := make(map[ResourceClass]ResourceConfig, len(managedResourceClasses))
	for _, class := range managedResourceClasses {
		resources[class] = ResourceConfig{
			MinAllocations: 0,
			MaxAllocations: 4,
			IdleTimeout:    Duration{30 * time.Minute},
			Walltime:       "08:00:00",
			Template:       "/opt/concord/" + string(class) + ".sbatch",
			WorkerEnvFile:  "/home/concord/.config/concord/" + string(class) + ".env",
			Partition:      "compute",
		}
	}
	return Config{
		Concord: ConcordConfig{
			BaseURL:        "https://concord.example",
			RequestTimeout: Duration{10 * time.Second},
			MaxSnapshotAge: Duration{10 * time.Minute},
			MaxFutureSkew:  Duration{30 * time.Second},
		},
		Slurm: SlurmConfig{
			SqueueBin:          "squeue",
			SacctBin:           "sacct",
			SbatchBin:          "sbatch",
			User:               "concord",
			Account:            "science",
			OwnerTag:           "concord-slurm-manager",
			LockFile:           "/tmp/concord-slurm-manager.lock",
			CommonEnvFile:      "/home/concord/.config/concord/common.env",
			CommandTimeout:     Duration{10 * time.Second},
			AccountingLookback: Duration{24 * time.Hour},
		},
		Resources: resources,
	}
}

func emptyDemandSnapshot(now time.Time) DemandSnapshot {
	resources := make([]ResourceDemand, 0, len(canonicalResourceClasses))
	for _, class := range canonicalResourceClasses {
		resources = append(resources, ResourceDemand{ResourceClass: class})
	}
	return DemandSnapshot{GeneratedAt: now, ResourceClasses: resources}
}

type fakeDemandSource struct {
	snapshot DemandSnapshot
	err      error
}

func (f fakeDemandSource) Fetch(context.Context) (DemandSnapshot, error) {
	return f.snapshot, f.err
}

type fakeSlurmClient struct {
	jobs         []ManagedJob
	listErr      error
	listErrors   []error
	submitErrors map[ResourceClass]error
	submissions  []ResourceClass
	listCalls    int
	nextID       int
}

func (f *fakeSlurmClient) ListManagedJobs(context.Context) ([]ManagedJob, error) {
	f.listCalls++
	if len(f.listErrors) > 0 {
		err := f.listErrors[0]
		f.listErrors = f.listErrors[1:]
		if err != nil {
			return nil, err
		}
	}
	if f.listErr != nil {
		return nil, f.listErr
	}
	return append([]ManagedJob(nil), f.jobs...), nil
}

func (f *fakeSlurmClient) Submit(_ context.Context, class ResourceClass, _ ResourceConfig) (string, error) {
	if err := f.submitErrors[class]; err != nil {
		return "", err
	}
	f.nextID++
	id := strconv.Itoa(f.nextID)
	f.submissions = append(f.submissions, class)
	f.jobs = append(f.jobs, ManagedJob{ID: id, ResourceClass: class, State: JobPending, SlurmState: "PENDING"})
	return id, nil
}

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
