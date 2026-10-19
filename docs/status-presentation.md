# Status presentation

The [main application status presentation reference](https://github.com/ChengLabResearch/ouroboros/blob/28999440891a91900f3a3703bb14614b506d74b6/documentation/docs/development/status-presentation.md)
defines the shared vocabulary, palette, checklist, and notice conventions.

The plugin's `src/statusPresentation.ts` maps existing observations at render time;
`StatusLabel` owns their labels and colors. These display values are not stored as
execution, connection, or readiness state. Page and its existing endpoints retain
those responsibilities.

The Rust startup endpoint's historical “Building Docker Image” step describes the
running backend service, so its row reads “Backend service.” Dev Docker Compose and
Container Startup rows retain their distinct meanings. The lazy ML-runtime warning
reads “Loads on first run” with neutral styling. Backend and browser volume checks
remain separate, explicitly named observations; a container running or a readiness
boolean does not establish a separately observed API connection.

For tasks, Page retains one `JobObservation` from the existing job-response flow.
`src/jobStatus.ts` derives stage labels and the overall result from that observation
at render time, never from a percentage. A reported failure survives dismissal of
its detailed diagnostic; a missing-job response retains any last progress while
showing an unknown outcome, even when no status snapshot preceded it. The existing
run/job-ID flags still schedule requests and provide pre-response waiting text.

The stateless `Notice` component owns message appearance, matching the main app's
`Alert`. Page retains the task-error list and its dismissal, cap, and deduplication;
model facilities retain their own nondismissible feedback. Existing errors keep
error severity and `role="alert"`, including communication/request errors, while
the reconnect message is informational. Severity does not infer job failure; the
separate source-derived task outcome remains visible after a diagnostic is dismissed.
