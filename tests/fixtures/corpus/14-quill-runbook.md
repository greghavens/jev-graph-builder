# Quill Runbook

## Overview

Quill is the document renderer. It is owned by the docs team. Quill stores its state in Redis.
Every write to Redis goes through a single connection pool with a fixed size.

## Details

When Redis becomes unavailable, Quill stops accepting new requests and returns a retryable error.
Operators restart Quill only after Redis reports healthy replication.
The docs team reviews every change to Quill before it is deployed.

## Notes

Quill depends on Atlas for service discovery. Configuration for Quill lives in the shared config repository.
