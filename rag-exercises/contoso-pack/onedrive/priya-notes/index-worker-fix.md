---
id: ONEDRIVE-priya-index-worker-fix
title: index worker fix
owner: Priya
modified: 2026-08-29
access: priya
class: Internal
platform: Content Hub
---

Note to self: if re-queue does not help for a tenant, stop the index worker, clear its temp folder, then start it and re-queue. Worked for Fabrikam Legal in July.

(the temp folder is on the app server, under the worker's data dir. took me two hours to find out. stop the worker FIRST or it locks the files.)

check afterwards: queue for the tenant goes down within ~15 min. if not, call the sub-team lead.
