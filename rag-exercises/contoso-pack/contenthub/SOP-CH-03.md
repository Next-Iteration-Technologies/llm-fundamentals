---
id: SOP-CH-03
title: Search issue triage
owner: Content Hub sub-team
revised: 2026-05-12
access: team
class: Confidential
platform: Content Hub
---

# SOP-CH-03 Search issue triage

## 1 Scope

This SOP covers reports that Content Hub search returns nothing, returns too little, or returns documents that cannot be opened. It applies to all tenants on Content Hub. For the Legacy Archive web viewer, use SOP-LA-02.

## 2 First checks

First checks when users report search problems: is it one tenant or all tenants? One document or all documents? Note the tenant and a sample document ID.

## 3 Index checks

Index checks: open the index queue for the tenant. If items are waiting for more than one hour, the index job for that tenant is stalled. Check the index worker status next.

## 4 Index workers

Restart the index worker and re-queue the folder.

## 5 Escalation

Escalation: if search still returns nothing after a re-index, raise a ticket to the Content Hub sub-team lead with the tenant, document ID and queue screenshot.
