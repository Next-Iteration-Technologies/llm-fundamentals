---
id: SOP-OPS-02
title: Log retention
owner: Operations
revised: 2025-11-20
access: team
class: Confidential
platform: All
---

# SOP-OPS-02 Log retention

## 1 Log retention

Server logs are kept for 14 days on each server, then deleted by the cleanup task.

## 2 Where the logs are

Each platform server writes its logs to its local log folder. Logs are read by logging into the server; there is no central log platform yet (see SD-LOG-01).

## 3 Requests for older logs

If someone needs logs older than the retention period, they are gone. Ask the requester whether the information can be found in tickets or alert mails instead.
