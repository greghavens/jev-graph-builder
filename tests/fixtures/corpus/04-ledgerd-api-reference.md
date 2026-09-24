# Ledgerd API reference

## Overview

Ledgerd is the billing ledger service. It is owned by the payments team. Ledgerd stores its state in Postgres.
Every write to Postgres goes through a single connection pool with a fixed size.

## Details

When Postgres becomes unavailable, Ledgerd stops accepting new requests and returns a retryable error.
Operators restart Ledgerd only after Postgres reports healthy replication.
The payments team reviews every change to Ledgerd before it is deployed.

## Notes

Ledgerd depends on Atlas for service discovery. Configuration for Ledgerd lives in the shared config repository.
