---
id: SOP-CAP-01
title: Capture remote site batch handling
owner: Capture sub-team
revised: 2026-07-30
access: team
class: Confidential
platform: Capture
---

# SOP-CAP-01 Capture remote site batch handling

## 1 Scope

Scanning batches from remote sites that do not arrive in Content Hub.

## 2 Find the batch

Look up the batch by its batch number in the Capture console. Note the site, the batch state and the time of the last state change.

## 3 Export failed

If the batch state is "export failed", the export to the target failed and was not retried (see KI-047). Retry the export from the console.

## 4 OCR failed

If pages show "OCR failed", check the scan quality. Blurry scans are sent back to the site for rescanning.
