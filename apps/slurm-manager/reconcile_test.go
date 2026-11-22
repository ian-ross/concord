package main

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestDesiredAllocationsUsesTasksAndBounds(t *testing.T) {
	tests := []struct {
		name     string
		demand   ResourceDemand
		config   ResourceConfig
		expected int
	}{
		{
			name:     "no demand",
			demand:   ResourceDemand{},
			config:   ResourceConfig{MinAllocations: 0, MaxAllocations: 5},
			expected: 0,
		},
		{
			name:     "queued and running tasks",
			demand:   ResourceDemand{Queued: 4, Running: 1},
			config:   ResourceConfig{MinAllocations: 0, MaxAllocations: 5},
			expected: 5,
		},
		{
			name:     "minimum",
			demand:   ResourceDemand{},
			config:   ResourceConfig{MinAllocations: 2, MaxAllocations: 5},
			expected: 2,
		},
		{
			name:     "maximum",
			demand:   ResourceDemand{Queued: 100},
			config:   ResourceConfig{MinAllocations: 0, MaxAllocations: 3},
			expected: 3,
		},
		{
			name:     "queued and running overflow",
			demand:   ResourceDemand{Queued: math.MaxInt64, Running: 1},
			config:   ResourceConfig{MinAllocations: 0, MaxAllocations: 4},
			expected: 4,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := desiredAllocations(test.demand, test.config); got != test.expected {
				t.Fatalf("desiredAllocations() = %d, want %d", got, test.expected)
			}
		})
	}
}

func TestActiveAllocationCountsIncludesOnlyPendingAndRunningForClass(t *testing.T) {
	pending, running := activeAllocationCounts([]ManagedJob{
		{ID: "pending", ResourceClass: ResourceCPU, State: JobPending},
		{ID: "running", ResourceClass: ResourceCPU, State: JobRunning},
		{ID: "held", ResourceClass: ResourceCPU, State: JobUnavailable},
		{ID: "terminal", ResourceClass: ResourceCPU, State: JobTerminal},
		{ID: "preempted", ResourceClass: ResourceCPU, State: JobPreempted},
		{ID: "gpu", ResourceClass: ResourceGPU, State: JobRunning},
	}, ResourceCPU)

	if pending != 1 || running != 1 {
		t.Fatalf("activeAllocationCounts() = pending %d running %d, want 1 and 1", pending, running)
	}
}

func TestReconcileSubmitsOnlyDeficitsForAllResourceClasses(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 3
	snapshot.ResourceClasses[1].Running = 1
	snapshot.ResourceClasses[2].Queued = 5
	snapshot.ResourceClasses[3].Queued = 1
	slurm := &fakeSlurmClient{jobs: []ManagedJob{
		{ID: "100", ResourceClass: ResourceCPU, State: JobRunning},
		{ID: "101", ResourceClass: ResourceGPU, State: JobPending},
	}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	want := []ResourceClass{ResourceCPU, ResourceCPU, ResourceCPUPreempt, ResourceGPU, ResourceGPU, ResourceGPU, ResourceGPUPreempt}
	if !reflect.DeepEqual(slurm.submissions, want) {
		t.Fatalf("submissions = %v, want %v", slurm.submissions, want)
	}
	if len(result.Classes) != 4 {
		t.Fatalf("class results = %d", len(result.Classes))
	}
}

func TestReconcileAppliesIndependentPerClassLimitsAndMinima(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	for index := 0; index < len(snapshot.ResourceClasses); index++ {
		snapshot.ResourceClasses[index].Queued = 100
	}
	config := validTestConfig()
	limits := map[ResourceClass]struct{ min, max int }{
		ResourceCPU:        {1, 1},
		ResourceCPUPreempt: {0, 0},
		ResourceGPU:        {0, 2},
		ResourceGPUPreempt: {1, 3},
	}
	for class, limit := range limits {
		resource := config.Resources[class]
		resource.MinAllocations = limit.min
		resource.MaxAllocations = limit.max
		config.Resources[class] = resource
	}
	slurm := &fakeSlurmClient{}
	reconciler := Reconciler{config: config, demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	if _, err := reconciler.Run(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	counts := make(map[ResourceClass]int)
	for _, class := range slurm.submissions {
		counts[class]++
	}
	want := map[ResourceClass]int{ResourceCPU: 1, ResourceGPU: 2, ResourceGPUPreempt: 3}
	if !reflect.DeepEqual(counts, want) {
		t.Fatalf("submission counts = %v, want %v", counts, want)
	}
}

func TestDryRunReportsDeficitWithoutSubmitting(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 2
	slurm := &fakeSlurmClient{}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), true)
	if err != nil {
		t.Fatal(err)
	}
	if len(slurm.submissions) != 0 || result.Classes[0].Deficit != 2 || !result.DryRun {
		t.Fatalf("unexpected dry-run result: submissions=%v result=%#v", slurm.submissions, result)
	}
	if slurm.listCalls != 1 {
		t.Fatalf("dry run listed managed jobs %d times, want initial list only", slurm.listCalls)
	}
}

func TestReconcileRelistsBeforeEachNonDryRunClass(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	slurm := &fakeSlurmClient{}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	if _, err := reconciler.Run(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	want := 1 + len(managedResourceClasses)
	if slurm.listCalls != want {
		t.Fatalf("managed job list calls = %d, want %d", slurm.listCalls, want)
	}
}

func TestRepeatedReconciliationIsDuplicateSafe(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 4
	slurm := &fakeSlurmClient{}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	if _, err := reconciler.Run(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	if _, err := reconciler.Run(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	if len(slurm.submissions) != 4 {
		t.Fatalf("submissions = %v, want exactly four from first pass", slurm.submissions)
	}
}

func TestHealthyExcessIsLeftForIdleDrain(t *testing.T) {
	now := time.Now().UTC()
	slurm := &fakeSlurmClient{jobs: []ManagedJob{
		{ID: "100", ResourceClass: ResourceCPU, State: JobRunning},
		{ID: "101", ResourceClass: ResourceCPU, State: JobPending},
	}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: emptyDemandSnapshot(now)}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(slurm.submissions) != 0 {
		t.Fatalf("unexpected submissions: %v", slurm.submissions)
	}
	if result.Classes[0].DesiredAllocations != 0 || result.Classes[0].RunningAllocations != 1 {
		t.Fatalf("unexpected CPU result: %#v", result.Classes[0])
	}
}

func TestHeldPendingCapacityIsReplaced(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 2
	slurm := &fakeSlurmClient{jobs: []ManagedJob{
		{ID: "held", ResourceClass: ResourceCPU, State: JobUnavailable, SlurmState: "PENDING", SlurmReason: "JobHeldUser"},
	}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	if _, err := reconciler.Run(context.Background(), false); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(slurm.submissions, []ResourceClass{ResourceCPU, ResourceCPU}) {
		t.Fatalf("submissions = %v", slurm.submissions)
	}
}

func TestPreemptedAndExpiredCapacityIsReplaced(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[1].Queued = 2
	snapshot.ResourceClasses[2].Queued = 2
	slurm := &fakeSlurmClient{jobs: []ManagedJob{
		{ID: "old-preempted", ResourceClass: ResourceCPUPreempt, State: JobPreempted, SlurmState: "PREEMPTED"},
		{ID: "old-expired", ResourceClass: ResourceGPU, State: JobTerminal, SlurmState: "TIMEOUT"},
	}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(slurm.submissions, []ResourceClass{ResourceCPUPreempt, ResourceCPUPreempt, ResourceGPU, ResourceGPU}) {
		t.Fatalf("submissions = %v", slurm.submissions)
	}
	if len(result.ManagedJobs) != 2 || result.ManagedJobs[0].State != JobPreempted || result.ManagedJobs[1].SlurmState != "TIMEOUT" {
		t.Fatalf("managed jobs = %#v", result.ManagedJobs)
	}
}

func TestRelistFailureDoesNotCorruptOtherClasses(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 1
	snapshot.ResourceClasses[2].Queued = 1
	slurm := &fakeSlurmClient{listErrors: []error{nil, errors.New("squeue unavailable")}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), false)
	if err == nil {
		t.Fatal("relist failure was not returned")
	}
	if !reflect.DeepEqual(slurm.submissions, []ResourceClass{ResourceGPU}) {
		t.Fatalf("submissions = %v, want GPU to continue", slurm.submissions)
	}
	if result.Classes[0].Error == "" || result.Classes[2].Error != "" {
		t.Fatalf("unexpected class errors: %#v", result.Classes)
	}
}

func TestSubmissionFailureDoesNotCorruptOtherClasses(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 2
	snapshot.ResourceClasses[2].Queued = 2
	slurm := &fakeSlurmClient{submitErrors: map[ResourceClass]error{ResourceCPU: errors.New("quota")}}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	result, err := reconciler.Run(context.Background(), false)
	if err == nil {
		t.Fatal("submission failure was not returned")
	}
	if !reflect.DeepEqual(slurm.submissions, []ResourceClass{ResourceGPU, ResourceGPU}) {
		t.Fatalf("submissions = %v, want GPU to continue", slurm.submissions)
	}
	if result.Classes[0].Error == "" || result.Classes[2].Error != "" {
		t.Fatalf("unexpected class errors: %#v", result.Classes)
	}
}

func TestInitialSlurmListFailureFailsClosedBeforeSubmission(t *testing.T) {
	now := time.Now().UTC()
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 2
	slurm := &fakeSlurmClient{listErr: errors.New("squeue unavailable")}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{snapshot: snapshot}, slurm: slurm, logger: testLogger()}

	if _, err := reconciler.Run(context.Background(), false); err == nil {
		t.Fatal("initial Slurm list failure ignored")
	}
	if len(slurm.submissions) != 0 {
		t.Fatalf("unexpected submissions after initial Slurm list failure: %v", slurm.submissions)
	}
}

func TestDemandFailureFailsClosedBeforeSlurmInspection(t *testing.T) {
	slurm := &fakeSlurmClient{}
	reconciler := Reconciler{config: validTestConfig(), demand: fakeDemandSource{err: errors.New("503")}, slurm: slurm, logger: testLogger()}
	if _, err := reconciler.Run(context.Background(), false); err == nil {
		t.Fatal("demand failure ignored")
	}
	if slurm.listCalls != 0 {
		t.Fatalf("Slurm inspected %d times after demand failure", slurm.listCalls)
	}
}
