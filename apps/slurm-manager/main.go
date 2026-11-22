package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
)

const allocationTokenEnv = "CONCORD_ALLOCATION_MANAGER_TOKEN"

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(arguments []string) int {
	flags := flag.NewFlagSet("concord-slurm-manager", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	configPath := flags.String("config", "", "path to allocation-manager JSON configuration")
	dryRun := flags.Bool("dry-run", false, "inspect and plan submissions without changing Slurm")
	status := flags.Bool("status", false, "emit a JSON status document without changing Slurm")
	if err := flags.Parse(arguments); err != nil {
		return 2
	}
	if *configPath == "" {
		fmt.Fprintln(os.Stderr, "--config is required")
		return 2
	}
	logger := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	config, err := loadConfig(*configPath)
	if err != nil {
		logger.Error("invalid allocation-manager configuration", "error", err)
		return 2
	}
	if err := validateRuntimeFiles(config); err != nil {
		logger.Error("unsafe allocation-manager runtime file", "error", err)
		return 2
	}
	lock, err := acquireProcessLock(config.Slurm.LockFile)
	if errors.Is(err, errLockHeld) {
		logger.Info("allocation-manager pass skipped because another process holds the lock")
		return 0
	}
	if err != nil {
		logger.Error("failed to acquire allocation-manager lock", "error", err)
		return 1
	}
	defer func() {
		if closeErr := lock.Close(); closeErr != nil {
			logger.Error("failed to release allocation-manager lock", "error", closeErr)
		}
	}()

	demandClient, err := newHTTPDemandClient(config.Concord, os.Getenv(allocationTokenEnv))
	if err != nil {
		logger.Error("invalid allocation-manager demand credentials", "error", err)
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	reconciler := Reconciler{
		config: config,
		demand: demandClient,
		slurm:  newCommandSlurmClient(config.Slurm, ExecCommandRunner{}, logger),
		logger: logger,
	}
	result, reconcileErr := reconciler.Run(ctx, *dryRun || *status)
	if *status {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(result); err != nil {
			logger.Error("failed to encode allocation-manager status", "error", err)
			return 1
		}
	}
	if reconcileErr != nil {
		logger.Error("allocation-manager reconciliation completed with errors", "error", reconcileErr)
		return 1
	}
	logger.Info("allocation-manager reconciliation completed", "dry_run", *dryRun || *status)
	return 0
}
