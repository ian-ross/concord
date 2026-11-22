package main

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

type commandCall struct {
	name string
	args []string
}

type fakeCommandRunner struct {
	outputs        map[string][]byte
	errors         map[string]error
	calls          []commandCall
	exportFileData []byte
}

func (f *fakeCommandRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	f.calls = append(f.calls, commandCall{name: name, args: append([]string(nil), args...)})
	for _, argument := range args {
		if path, found := strings.CutPrefix(argument, "--export-file="); found {
			f.exportFileData, _ = os.ReadFile(path)
		}
	}
	if err := f.errors[name]; err != nil {
		return nil, err
	}
	return f.outputs[name], nil
}

func TestSlurmSubprocessEnvironmentOmitsAllocationToken(t *testing.T) {
	environment := []string{"PATH=/usr/bin", allocationTokenEnv + "=secret", "SLURM_CONF=/etc/slurm.conf"}
	got := environmentWithoutAllocationToken(environment)
	want := []string{"PATH=/usr/bin", "SLURM_CONF=/etc/slurm.conf"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("filtered environment = %v, want %v", got, want)
	}
}

func TestSlurmClientDiscoversOnlyOwnedJobsAndNormalizesStates(t *testing.T) {
	runner := &fakeCommandRunner{outputs: map[string][]byte{
		"squeue": []byte(strings.Join([]string{
			"100|PENDING|Resources|concord-slurm-manager:cpu",
			"101|RUNNING|None|concord-slurm-manager:gpu",
			"102|SUSPENDED|None|concord-slurm-manager:cpu-preempt",
			"103|COMPLETING|None|concord-slurm-manager:gpu-preempt",
			"104|PENDING|JobHeldUser|concord-slurm-manager:cpu",
			"999|RUNNING|None|someone-else:cpu",
		}, "\n")),
		"sacct": []byte(strings.Join([]string{
			"97|STAGE_OUT|None|concord-slurm-manager:cpu",
			"98|PREEMPTED|Priority|concord-slurm-manager:cpu-preempt",
			"98.batch|FAILED|None|concord-slurm-manager:cpu-preempt",
			"99|TIMEOUT+|None|concord-slurm-manager:gpu-preempt",
			"101|RUNNING|None|concord-slurm-manager:gpu",
		}, "\n")),
	}}
	config := validTestConfig().Slurm
	client := newCommandSlurmClient(config, runner, testLogger())
	client.now = func() time.Time { return time.Date(2026, 8, 11, 12, 0, 0, 0, time.UTC) }

	jobs, err := client.ListManagedJobs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	want := []ManagedJob{
		{ID: "100", ResourceClass: ResourceCPU, State: JobPending, SlurmState: "PENDING", SlurmReason: "Resources"},
		{ID: "101", ResourceClass: ResourceGPU, State: JobRunning, SlurmState: "RUNNING", SlurmReason: "None"},
		{ID: "102", ResourceClass: ResourceCPUPreempt, State: JobUnavailable, SlurmState: "SUSPENDED", SlurmReason: "None"},
		{ID: "103", ResourceClass: ResourceGPUPreempt, State: JobUnavailable, SlurmState: "COMPLETING", SlurmReason: "None"},
		{ID: "104", ResourceClass: ResourceCPU, State: JobUnavailable, SlurmState: "PENDING", SlurmReason: "JobHeldUser"},
		{ID: "97", ResourceClass: ResourceCPU, State: JobTerminal, SlurmState: "STAGE_OUT", SlurmReason: "None"},
		{ID: "98", ResourceClass: ResourceCPUPreempt, State: JobPreempted, SlurmState: "PREEMPTED", SlurmReason: "Priority"},
		{ID: "99", ResourceClass: ResourceGPUPreempt, State: JobTerminal, SlurmState: "TIMEOUT", SlurmReason: "None"},
	}
	if !reflect.DeepEqual(jobs, want) {
		t.Fatalf("jobs = %#v, want %#v", jobs, want)
	}
}

func TestSlurmClientCanContinueWhenOptionalAccountingIsUnavailable(t *testing.T) {
	runner := &fakeCommandRunner{
		outputs: map[string][]byte{"squeue": []byte("100|RUNNING|None|concord-slurm-manager:cpu\n")},
		errors:  map[string]error{"sacct": errors.New("accounting disabled")},
	}
	client := newCommandSlurmClient(validTestConfig().Slurm, runner, testLogger())

	jobs, err := client.ListManagedJobs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 1 || jobs[0].ID != "100" {
		t.Fatalf("active jobs = %#v", jobs)
	}

	config := validTestConfig().Slurm
	config.AccountingRequired = true
	client = newCommandSlurmClient(config, runner, testLogger())
	if _, err := client.ListManagedJobs(context.Background()); err == nil {
		t.Fatal("required accounting failure ignored")
	}
}

func TestSlurmClientRejectsUnknownOwnedState(t *testing.T) {
	runner := &fakeCommandRunner{outputs: map[string][]byte{
		"squeue": []byte("100|MYSTERY|None|concord-slurm-manager:cpu\n"),
	}}
	client := newCommandSlurmClient(validTestConfig().Slurm, runner, testLogger())
	if _, err := client.ListManagedJobs(context.Background()); err == nil || !strings.Contains(err.Error(), "unknown Slurm state") {
		t.Fatalf("error = %v", err)
	}
}

func TestSlurmSubmissionUsesOwnershipAndProtectedPathsWithoutSecrets(t *testing.T) {
	runner := &fakeCommandRunner{outputs: map[string][]byte{"sbatch": []byte("12345;cluster\n")}}
	config := validTestConfig()
	resource := config.Resources[ResourceGPUPreempt]
	resource.QOS = "preempt"
	client := newCommandSlurmClient(config.Slurm, runner, testLogger())

	jobID, err := client.Submit(context.Background(), ResourceGPUPreempt, resource)
	if err != nil {
		t.Fatal(err)
	}
	if jobID != "12345" {
		t.Fatalf("job ID = %q", jobID)
	}
	call := runner.calls[0]
	joined := strings.Join(call.args, " ")
	for _, expected := range []string{
		"--comment=concord-slurm-manager:gpu-preempt",
		"--partition=compute",
		"--qos=preempt",
		"--account=science",
		"--export-file=",
		"/opt/concord/gpu-preempt.sbatch",
	} {
		if !strings.Contains(joined, expected) {
			t.Errorf("sbatch args %q missing %q", joined, expected)
		}
	}
	if strings.Contains(joined, "TOKEN") {
		t.Fatalf("sbatch args contain token material: %q", joined)
	}
	wantExport := "CONCORD_WORKER_COMMON_ENV_FILE=/home/concord/.config/concord/common.env\x00" +
		"CONCORD_WORKER_ENV_FILE=/home/concord/.config/concord/gpu-preempt.env\x00" +
		"CONCORD_WORKER_IDLE_TIMEOUT_SECONDS=1800\x00"
	if string(runner.exportFileData) != wantExport {
		t.Fatalf("export-file data = %q, want %q", runner.exportFileData, wantExport)
	}
}

func TestSlurmSubmissionFailureIsReturned(t *testing.T) {
	runner := &fakeCommandRunner{errors: map[string]error{"sbatch": errors.New("quota")}}
	config := validTestConfig()
	client := newCommandSlurmClient(config.Slurm, runner, testLogger())
	if _, err := client.Submit(context.Background(), ResourceCPU, config.Resources[ResourceCPU]); err == nil {
		t.Fatal("sbatch failure ignored")
	}
}
