package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type ResourceClass string

const (
	ResourceCPU        ResourceClass = "cpu"
	ResourceCPUPreempt ResourceClass = "cpu-preempt"
	ResourceGPU        ResourceClass = "gpu"
	ResourceGPUPreempt ResourceClass = "gpu-preempt"
)

var canonicalResourceClasses = [...]ResourceClass{
	ResourceCPU,
	ResourceCPUPreempt,
	ResourceGPU,
	ResourceGPUPreempt,
}

var managedResourceClasses = [...]ResourceClass{
	ResourceCPU,
	ResourceCPUPreempt,
	ResourceGPU,
	ResourceGPUPreempt,
}

type Duration struct {
	time.Duration
}

func (d *Duration) UnmarshalJSON(data []byte) error {
	var value string
	if err := json.Unmarshal(data, &value); err != nil {
		return errors.New("duration must be a string")
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", value, err)
	}
	d.Duration = parsed
	return nil
}

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

type Config struct {
	Concord   ConcordConfig                    `json:"concord"`
	Slurm     SlurmConfig                      `json:"slurm"`
	Resources map[ResourceClass]ResourceConfig `json:"resources"`
}

type ConcordConfig struct {
	BaseURL        string   `json:"base_url"`
	RequestTimeout Duration `json:"request_timeout"`
	MaxSnapshotAge Duration `json:"max_snapshot_age"`
	MaxFutureSkew  Duration `json:"max_future_skew"`
}

type SlurmConfig struct {
	SqueueBin          string   `json:"squeue_bin"`
	SacctBin           string   `json:"sacct_bin"`
	SbatchBin          string   `json:"sbatch_bin"`
	User               string   `json:"user"`
	Account            string   `json:"account"`
	OwnerTag           string   `json:"owner_tag"`
	LockFile           string   `json:"lock_file"`
	CommonEnvFile      string   `json:"common_env_file"`
	CommandTimeout     Duration `json:"command_timeout"`
	AccountingLookback Duration `json:"accounting_lookback"`
	AccountingRequired bool     `json:"accounting_required"`
}

type ResourceConfig struct {
	MinAllocations int      `json:"min_allocations"`
	MaxAllocations int      `json:"max_allocations"`
	IdleTimeout    Duration `json:"idle_timeout"`
	Walltime       string   `json:"walltime"`
	Template       string   `json:"template"`
	WorkerEnvFile  string   `json:"worker_env_file"`
	Partition      string   `json:"partition"`
	QOS            string   `json:"qos"`
}

func loadConfig(path string) (Config, error) {
	if err := validateRuntimeFile(path, false); err != nil {
		return Config{}, fmt.Errorf("config file: %w", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read config: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var config Config
	if err := decoder.Decode(&config); err != nil {
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return Config{}, errors.New("decode config: multiple JSON values")
		}
		return Config{}, fmt.Errorf("decode config: %w", err)
	}
	if err := config.validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

func (c Config) validate() error {
	if err := c.Concord.validate(); err != nil {
		return fmt.Errorf("concord config: %w", err)
	}
	if err := c.Slurm.validate(); err != nil {
		return fmt.Errorf("slurm config: %w", err)
	}
	if len(c.Resources) != len(managedResourceClasses) {
		return fmt.Errorf("resources must configure exactly cpu, cpu-preempt, gpu, and gpu-preempt")
	}
	for _, class := range managedResourceClasses {
		resource, ok := c.Resources[class]
		if !ok {
			return fmt.Errorf("resources missing %q", class)
		}
		if err := resource.validate(); err != nil {
			return fmt.Errorf("resource %q: %w", class, err)
		}
	}
	for class := range c.Resources {
		if !isManagedResourceClass(class) {
			return fmt.Errorf("unsupported managed resource class %q", class)
		}
	}
	return nil
}

func (c ConcordConfig) validate() error {
	parsed, err := url.Parse(c.BaseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return errors.New("base_url must be an absolute HTTP(S) URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return errors.New("base_url must use http or https")
	}
	if parsed.Scheme == "http" && !isLoopbackHost(parsed.Hostname()) {
		return errors.New("base_url must use https except for loopback testing")
	}
	if parsed.User != nil {
		return errors.New("base_url must not contain credentials")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || (parsed.Path != "" && parsed.Path != "/") {
		return errors.New("base_url must not contain a path, query, or fragment")
	}
	if c.RequestTimeout.Duration <= 0 {
		return errors.New("request_timeout must be positive")
	}
	if c.MaxSnapshotAge.Duration <= 0 {
		return errors.New("max_snapshot_age must be positive")
	}
	if c.MaxFutureSkew.Duration < 0 {
		return errors.New("max_future_skew must not be negative")
	}
	return nil
}

func (c SlurmConfig) validate() error {
	for name, value := range map[string]string{
		"squeue_bin":      c.SqueueBin,
		"sacct_bin":       c.SacctBin,
		"sbatch_bin":      c.SbatchBin,
		"user":            c.User,
		"owner_tag":       c.OwnerTag,
		"lock_file":       c.LockFile,
		"common_env_file": c.CommonEnvFile,
	} {
		if err := validateArgument(name, value); err != nil {
			return err
		}
	}
	if c.Account != "" {
		if err := validateArgument("account", c.Account); err != nil {
			return err
		}
	}
	if strings.ContainsAny(c.OwnerTag, ":|") {
		return errors.New("owner_tag must not contain ':' or '|'")
	}
	if !filepath.IsAbs(c.LockFile) {
		return errors.New("lock_file must be absolute")
	}
	if !filepath.IsAbs(c.CommonEnvFile) {
		return errors.New("common_env_file must be absolute")
	}
	if c.CommandTimeout.Duration <= 0 {
		return errors.New("command_timeout must be positive")
	}
	if c.AccountingLookback.Duration <= 0 {
		return errors.New("accounting_lookback must be positive")
	}
	return nil
}

func (c ResourceConfig) validate() error {
	if c.MinAllocations < 0 {
		return errors.New("min_allocations must not be negative")
	}
	if c.MaxAllocations < c.MinAllocations {
		return errors.New("max_allocations must be at least min_allocations")
	}
	if c.IdleTimeout.Duration < time.Second || c.IdleTimeout.Duration%time.Second != 0 {
		return errors.New("idle_timeout must be a positive whole number of seconds")
	}
	for name, value := range map[string]string{
		"walltime":        c.Walltime,
		"template":        c.Template,
		"worker_env_file": c.WorkerEnvFile,
		"partition":       c.Partition,
	} {
		if err := validateArgument(name, value); err != nil {
			return err
		}
	}
	if c.QOS != "" {
		if err := validateArgument("qos", c.QOS); err != nil {
			return err
		}
	}
	if !filepath.IsAbs(c.Template) {
		return errors.New("template must be absolute")
	}
	if !filepath.IsAbs(c.WorkerEnvFile) {
		return errors.New("worker_env_file must be absolute")
	}
	return nil
}

func validateArgument(name, value string) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s must not be blank", name)
	}
	if value != strings.TrimSpace(value) {
		return fmt.Errorf("%s must not have surrounding whitespace", name)
	}
	if strings.ContainsAny(value, "\x00\r\n,|") {
		return fmt.Errorf("%s contains an unsupported delimiter", name)
	}
	return nil
}

func validateRuntimeFiles(config Config) error {
	if err := validateRuntimeFile(config.Slurm.CommonEnvFile, true); err != nil {
		return fmt.Errorf("common_env_file: %w", err)
	}
	for _, class := range managedResourceClasses {
		resource := config.Resources[class]
		if err := validateRuntimeFile(resource.WorkerEnvFile, true); err != nil {
			return fmt.Errorf("resource %q worker_env_file: %w", class, err)
		}
		if err := validateRuntimeFile(resource.Template, false); err != nil {
			return fmt.Errorf("resource %q template: %w", class, err)
		}
	}
	return nil
}

func validateRuntimeFile(path string, protected bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("must be a regular file, not a symlink or special file")
	}
	permissions := info.Mode().Perm()
	if permissions&0o022 != 0 {
		return errors.New("must not be group/world writable")
	}
	if protected && permissions&0o077 != 0 {
		return errors.New("must not be group/world accessible")
	}
	return nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	address := net.ParseIP(host)
	return address != nil && address.IsLoopback()
}

func isManagedResourceClass(class ResourceClass) bool {
	for _, candidate := range managedResourceClasses {
		if class == candidate {
			return true
		}
	}
	return false
}
