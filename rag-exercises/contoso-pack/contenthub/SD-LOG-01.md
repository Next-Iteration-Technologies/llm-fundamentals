---
id: SD-LOG-01
title: Central log platform, solution design (draft)
owner: Sam
revised: 2026-09-10
access: team
class: Confidential
platform: POC
stream: POC central log platform
---

# SD-LOG-01 Central log platform, solution design (draft)

## 1 Problem

Logs can only be read by logging into each server. Correlating an alert mail with transfer.log and index.log across servers takes the on-call engineer a long time, and logs older than the retention period are lost.

## 2 Requirements

Requirements for the central log platform: logs can carry user and document identifiers, so collection and parsing must run on-premise. Search within 5 seconds.

## 3 Scope of the first phase

Content Hub server logs first (transfer.log, index.log, render.log). Legacy Archive and Capture follow once the platform is chosen.

## 4 Candidates

Vendor A, vendor B and vendor C, evaluated in the POC (see the POC minutes).

## 5 Open points

Retention on the platform; alert rules for known patterns (KI-022 transfer timeouts first); access for the Content Hub sub-team.
