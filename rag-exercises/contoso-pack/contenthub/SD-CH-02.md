---
id: SD-CH-02
title: Content Hub indexing, design overview
owner: Content Hub sub-team
revised: 2026-02-20
access: team
class: Confidential
platform: Content Hub
---

# SD-CH-02 Content Hub indexing, design overview

## 1 Flow

A document arrives at stage transfer, is written to the tenant's folder on the target, and is then queued for indexing. The index worker reads the queue, extracts text and updates the tenant's search index.

## 2 One queue per tenant

Each tenant has its own index queue and index job. A stalled job affects one tenant only, which is why "one tenant or all tenants" is the first question in SOP-CH-03.

## 3 Index workers

Index workers run on each Content Hub application server. A worker keeps extracted text in its temp folder until the index is updated.

## 4 Night batch

Bulk transfers run between 22:00 and 06:00. During the batch the target is often busy; transfers retry three times before they fail with TIMEOUT.
