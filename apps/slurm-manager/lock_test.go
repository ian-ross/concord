package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestProcessLockRejectsSymlink(t *testing.T) {
	directory := t.TempDir()
	target := filepath.Join(directory, "target")
	path := filepath.Join(directory, "manager.lock")
	if err := os.WriteFile(target, []byte("preserve"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, path); err != nil {
		t.Fatal(err)
	}

	if lock, err := acquireProcessLock(path); err == nil {
		_ = lock.Close()
		t.Fatal("symlink lock path accepted")
	}
	contents, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(contents) != "preserve" {
		t.Fatalf("symlink target changed to %q", contents)
	}
}

func TestProcessLockRejectsConcurrentPass(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manager.lock")
	first, err := acquireProcessLock(path)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()

	if second, err := acquireProcessLock(path); !errors.Is(err, errLockHeld) {
		if err == nil {
			_ = second.Close()
		}
		t.Fatalf("second lock error = %v, want errLockHeld", err)
	}
}
