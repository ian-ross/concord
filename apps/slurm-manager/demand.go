package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	demandPath        = "/api/internal/worker-demand"
	maxDemandBodySize = 1 << 20
)

type ResourceDemand struct {
	ResourceClass  ResourceClass `json:"resource_class"`
	Queued         int64         `json:"queued"`
	Running        int64         `json:"running"`
	WorkerCapacity int64         `json:"worker_capacity"`
}

type DemandSnapshot struct {
	GeneratedAt     time.Time        `json:"generated_at"`
	ResourceClasses []ResourceDemand `json:"resource_classes"`
}

func (s DemandSnapshot) byResource() map[ResourceClass]ResourceDemand {
	result := make(map[ResourceClass]ResourceDemand, len(s.ResourceClasses))
	for _, resource := range s.ResourceClasses {
		result[resource.ResourceClass] = resource
	}
	return result
}

type DemandSource interface {
	Fetch(context.Context) (DemandSnapshot, error)
}

type HTTPDemandClient struct {
	endpoint       string
	token          string
	client         *http.Client
	maxSnapshotAge time.Duration
	maxFutureSkew  time.Duration
	now            func() time.Time
}

func newHTTPDemandClient(config ConcordConfig, token string) (*HTTPDemandClient, error) {
	if err := validateToken(token); err != nil {
		return nil, err
	}
	base, err := url.Parse(config.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse Concord base URL: %w", err)
	}
	base.Path = demandPath
	return &HTTPDemandClient{
		endpoint: base.String(),
		token:    token,
		client: &http.Client{
			Timeout: config.RequestTimeout.Duration,
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		maxSnapshotAge: config.MaxSnapshotAge.Duration,
		maxFutureSkew:  config.MaxFutureSkew.Duration,
		now:            time.Now,
	}, nil
}

// Fetch reads one worker-demand snapshot from Concord.
//
// Responsibilities:
//   - Send the allocation-manager bearer token only to the configured Concord
//     demand endpoint.
//   - Decode one strict JSON snapshot and validate it before returning it to the
//     reconciler.
//
// Rules:
//   - Fail closed on transport, authentication, schema, freshness, future-skew,
//     incomplete or non-canonical class, and negative count failures.
func (c *HTTPDemandClient) Fetch(ctx context.Context) (DemandSnapshot, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint, nil)
	if err != nil {
		return DemandSnapshot{}, fmt.Errorf("create demand request: %w", err)
	}
	request.Header.Set("Authorization", "Bearer "+c.token)
	request.Header.Set("Accept", "application/json")
	response, err := c.client.Do(request)
	if err != nil {
		return DemandSnapshot{}, fmt.Errorf("fetch worker demand: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, maxDemandBodySize))
		return DemandSnapshot{}, fmt.Errorf("fetch worker demand: Concord returned HTTP %d", response.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(response.Body, maxDemandBodySize+1))
	if err != nil {
		return DemandSnapshot{}, fmt.Errorf("read worker demand: %w", err)
	}
	if len(body) > maxDemandBodySize {
		return DemandSnapshot{}, errors.New("worker demand response exceeds 1 MiB")
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var snapshot DemandSnapshot
	if err := decoder.Decode(&snapshot); err != nil {
		return DemandSnapshot{}, fmt.Errorf("decode worker demand: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return DemandSnapshot{}, errors.New("decode worker demand: multiple JSON values")
		}
		return DemandSnapshot{}, fmt.Errorf("decode worker demand: %w", err)
	}
	if err := validateDemandSnapshot(snapshot, c.now(), c.maxSnapshotAge, c.maxFutureSkew); err != nil {
		return DemandSnapshot{}, err
	}
	return snapshot, nil
}

func validateDemandSnapshot(snapshot DemandSnapshot, now time.Time, maxAge, maxFutureSkew time.Duration) error {
	if snapshot.GeneratedAt.IsZero() {
		return errors.New("worker demand generated_at is required")
	}
	_, offset := snapshot.GeneratedAt.Zone()
	if offset != 0 {
		return errors.New("worker demand generated_at must be UTC")
	}
	age := now.Sub(snapshot.GeneratedAt)
	if age > maxAge {
		return fmt.Errorf("worker demand snapshot is stale: age %s exceeds %s", age.Round(time.Second), maxAge)
	}
	if age < -maxFutureSkew {
		return fmt.Errorf("worker demand snapshot is too far in the future: skew %s exceeds %s", (-age).Round(time.Second), maxFutureSkew)
	}
	if len(snapshot.ResourceClasses) != len(canonicalResourceClasses) {
		return errors.New("worker demand must contain all four canonical resource classes exactly once")
	}
	for index, expected := range canonicalResourceClasses {
		resource := snapshot.ResourceClasses[index]
		if resource.ResourceClass != expected {
			return fmt.Errorf("worker demand resource_classes[%d] must be %q", index, expected)
		}
		if resource.Queued < 0 || resource.Running < 0 || resource.WorkerCapacity < 0 {
			return fmt.Errorf("worker demand for %q contains a negative count", expected)
		}
	}
	return nil
}

func validateToken(token string) error {
	if strings.TrimSpace(token) == "" {
		return errors.New("CONCORD_ALLOCATION_MANAGER_TOKEN must be non-blank")
	}
	for _, value := range []byte(token) {
		if value < 0x21 || value > 0x7e {
			return errors.New("CONCORD_ALLOCATION_MANAGER_TOKEN must contain printable ASCII without whitespace")
		}
	}
	return nil
}
