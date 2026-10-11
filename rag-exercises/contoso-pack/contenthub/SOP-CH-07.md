---
id: SOP-CH-07
title: Alert mail handling
owner: Content Hub sub-team
revised: 2026-03-02
access: team
class: Confidential
platform: Content Hub
---

# SOP-CH-07 Alert mail handling

## 1 Scope

Content Hub sends an alert mail to archive-ops@contoso.example when a document fails a processing stage (transfer, index, render). The subject carries the document ID, the stage and the result.

## 2 Triage the mailbox

Group the alert mails of the night by stage and result. Many alerts for one stage within a few minutes usually share one cause; handle them as one incident and note all document IDs.

## 3 Read the logs

For each alert mail, take the document ID and read transfer.log and index.log for that ID. A TIMEOUT at stage transfer during the night batch usually means the target was busy.

## 4 Reprocess

Reprocess failed document IDs after 06:00, when the night batch has finished. Do not reprocess during the batch window (22:00 to 06:00); it adds load to the busy target.

## 5 Close

Reply to the alert mail with "reprocessed" and the time, so the next on-call engineer sees it is handled. If the same document ID fails twice, open a ticket and link the known issue.
