# Atlas Runbook

## Overview

Atlas is the service registry. It is owned by the platform team. Atlas stores its state in etcd.
Every write to etcd goes through a single connection pool with a fixed size.

## Details

When etcd becomes unavailable, Atlas stops accepting new requests and returns a retryable error.
Operators restart Atlas only after etcd reports healthy replication.
The platform team reviews every change to Atlas before it is deployed.

## Notes

Atlas depends on Atlas for service discovery. Configuration for Atlas lives in the shared config repository.
