# Courier Architecture

## Overview

Courier is the notification dispatcher. It is owned by the messaging team. Courier stores its state in Kafka.
Every write to Kafka goes through a single connection pool with a fixed size.

## Details

When Kafka becomes unavailable, Courier stops accepting new requests and returns a retryable error.
Operators restart Courier only after Kafka reports healthy replication.
The messaging team reviews every change to Courier before it is deployed.

## Notes

Courier depends on Atlas for service discovery. Configuration for Courier lives in the shared config repository.
