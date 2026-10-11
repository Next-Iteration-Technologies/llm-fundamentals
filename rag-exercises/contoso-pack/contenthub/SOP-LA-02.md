---
id: SOP-LA-02
title: Legacy Archive web viewer recovery
owner: Legacy Archive sub-team
revised: 2025-12-08
access: team
class: Confidential
platform: Legacy Archive
---

# SOP-LA-02 Legacy Archive web viewer recovery

## 1 Scope

Users report that documents do not open in the Legacy Archive web viewer.

## 2 Check the API

Open a document by its ID through the API test page. If the API returns the document, the archive is fine and the problem is the viewer.

## 3 Check the sessions

Read the session count on the viewer's status page. At the session limit, new users get an empty page (see KI-031).

## 4 Recycle

Recycle the viewer's app pool outside business hours if possible. Tell the users that open sessions will be lost.
