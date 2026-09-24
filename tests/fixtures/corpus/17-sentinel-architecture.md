# Sentinel Architecture

## Overview

Sentinel is the alerting pipeline. It is owned by the observability team. Sentinel stores its state in Prometheus.
Every write to Prometheus goes through a single connection pool with a fixed size.

## Details

When Prometheus becomes unavailable, Sentinel stops accepting new requests and returns a retryable error.
Operators restart Sentinel only after Prometheus reports healthy replication.
The observability team reviews every change to Sentinel before it is deployed.

## Notes

Sentinel depends on Atlas for service discovery. Configuration for Sentinel lives in the shared config repository.
