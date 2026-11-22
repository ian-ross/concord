package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigValidatesExactlyFourManagedClasses(t *testing.T) {
	config := validTestConfig()
	if err := config.validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}

	delete(config.Resources, ResourceGPUPreempt)
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "exactly") {
		t.Fatalf("missing resource error = %v", err)
	}

	config = validTestConfig()
	config.Resources[ResourceClass("server")] = config.Resources[ResourceCPU]
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "exactly") {
		t.Fatalf("extra server resource error = %v", err)
	}
}

func TestConfigRejectsPlaintextRemoteConcordAndWhitespaceOwnership(t *testing.T) {
	config := validTestConfig()
	config.Concord.BaseURL = "http://concord.example"
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("plaintext remote URL error = %v", err)
	}

	config = validTestConfig()
	config.Slurm.OwnerTag = " concord-slurm-manager"
	if err := config.validate(); err == nil || !strings.Contains(err.Error(), "whitespace") {
		t.Fatalf("owner-tag whitespace error = %v", err)
	}
}

func TestConfigRejectsInvalidBoundsAndIdleTimeout(t *testing.T) {
	tests := []struct {
		name   string
		change func(*ResourceConfig)
		want   string
	}{
		{"negative minimum", func(c *ResourceConfig) { c.MinAllocations = -1 }, "min_allocations"},
		{"maximum below minimum", func(c *ResourceConfig) { c.MinAllocations = 2; c.MaxAllocations = 1 }, "max_allocations"},
		{"zero idle timeout", func(c *ResourceConfig) { c.IdleTimeout.Duration = 0 }, "idle_timeout"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := validTestConfig()
			resource := config.Resources[ResourceCPU]
			test.change(&resource)
			config.Resources[ResourceCPU] = resource
			err := config.validate()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validation error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestRuntimeFilesMustBeRegularAndProtected(t *testing.T) {
	directory := t.TempDir()
	config := validTestConfig()
	common := filepath.Join(directory, "common.env")
	if err := os.WriteFile(common, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	config.Slurm.CommonEnvFile = common
	for _, class := range managedResourceClasses {
		resource := config.Resources[class]
		resource.WorkerEnvFile = filepath.Join(directory, string(class)+".env")
		resource.Template = filepath.Join(directory, string(class)+".sbatch")
		if err := os.WriteFile(resource.WorkerEnvFile, []byte("secret"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(resource.Template, []byte("#!/bin/bash\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		config.Resources[class] = resource
	}
	if err := validateRuntimeFiles(config); err != nil {
		t.Fatalf("safe runtime files rejected: %v", err)
	}

	if err := os.Chmod(common, 0o640); err != nil {
		t.Fatal(err)
	}
	if err := validateRuntimeFiles(config); err == nil || !strings.Contains(err.Error(), "accessible") {
		t.Fatalf("readable protected file error = %v", err)
	}
}

func TestDeploymentConfigExampleIsValid(t *testing.T) {
	path := filepath.Join("..", "..", "deploy", "slurm-manager", "config.json.example")
	if _, err := loadConfig(path); err != nil {
		t.Fatalf("deployment config example is invalid: %v", err)
	}
}

func TestLoadConfigRejectsUnknownAndTrailingJSON(t *testing.T) {
	directory := t.TempDir()
	for name, contents := range map[string]string{
		"unknown":  `{"unexpected": true}`,
		"trailing": `{}` + "\n{}",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(directory, name+".json")
			if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := loadConfig(path); err == nil {
				t.Fatal("invalid JSON config accepted")
			}
		})
	}
}
