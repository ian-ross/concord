package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHTTPDemandClientAuthenticatesAndValidatesAllClasses(t *testing.T) {
	now := time.Date(2026, 8, 11, 12, 0, 0, 0, time.UTC)
	snapshot := emptyDemandSnapshot(now)
	snapshot.ResourceClasses[0].Queued = 3
	snapshot.ResourceClasses[2].Running = 1
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != demandPath {
			t.Errorf("path = %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer allocation-secret" {
			t.Errorf("authorization header missing")
		}
		_ = json.NewEncoder(writer).Encode(snapshot)
	}))
	defer server.Close()
	config := validTestConfig().Concord
	config.BaseURL = server.URL
	client, err := newHTTPDemandClient(config, "allocation-secret")
	if err != nil {
		t.Fatal(err)
	}
	client.now = func() time.Time { return now }

	got, err := client.Fetch(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got.ResourceClasses[0].Queued != 3 || got.ResourceClasses[2].Running != 1 {
		t.Fatalf("unexpected snapshot: %#v", got)
	}
}

func TestHTTPDemandClientFailsClosed(t *testing.T) {
	now := time.Date(2026, 8, 11, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name     string
		response func(http.ResponseWriter)
		want     string
	}{
		{
			name: "unavailable",
			response: func(writer http.ResponseWriter) {
				writer.WriteHeader(http.StatusServiceUnavailable)
				_, _ = writer.Write([]byte(`{"detail":"unavailable"}`))
			},
			want: "HTTP 503",
		},
		{
			name: "stale",
			response: func(writer http.ResponseWriter) {
				snapshot := emptyDemandSnapshot(now.Add(-11 * time.Minute))
				_ = json.NewEncoder(writer).Encode(snapshot)
			},
			want: "stale",
		},
		{
			name: "missing class",
			response: func(writer http.ResponseWriter) {
				snapshot := emptyDemandSnapshot(now)
				snapshot.ResourceClasses = snapshot.ResourceClasses[:3]
				_ = json.NewEncoder(writer).Encode(snapshot)
			},
			want: "all four",
		},
		{
			name: "wrong order",
			response: func(writer http.ResponseWriter) {
				snapshot := emptyDemandSnapshot(now)
				snapshot.ResourceClasses[0], snapshot.ResourceClasses[1] = snapshot.ResourceClasses[1], snapshot.ResourceClasses[0]
				_ = json.NewEncoder(writer).Encode(snapshot)
			},
			want: "resource_classes[0]",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				test.response(writer)
			}))
			defer server.Close()
			config := validTestConfig().Concord
			config.BaseURL = server.URL
			client, err := newHTTPDemandClient(config, "allocation-secret")
			if err != nil {
				t.Fatal(err)
			}
			client.now = func() time.Time { return now }
			_, err = client.Fetch(context.Background())
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Fetch error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestAllocationTokenMustBePrintableASCIIWithoutWhitespace(t *testing.T) {
	for _, token := range []string{"", "  ", "has space", "tökén", "line\nbreak"} {
		if err := validateToken(token); err == nil {
			t.Errorf("token %q accepted", token)
		}
	}
	if err := validateToken("safe-token_123"); err != nil {
		t.Fatalf("valid token rejected: %v", err)
	}
}
