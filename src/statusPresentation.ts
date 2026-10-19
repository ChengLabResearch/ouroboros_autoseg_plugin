// Display vocabulary only; observations stay with Page and the existing endpoints.
// Shared reference: docs/status-presentation.md.
export const STATUS_LABELS = {
    waiting: 'Waiting',
    running: 'Running',
    complete: 'Complete',
    warning: 'Warning',
    failed: 'Failed',
    interrupted: 'Interrupted',
    unknown: 'Unknown',
} as const;

export type PresentationStatus = keyof typeof STATUS_LABELS;
export type BackendStep = { name: string; status: string; };
export type BackendStatus = {
    is_ready: boolean;
    initialization_steps: BackendStep[];
    start_time: number | null;
    ready_time: number | null;
};
export type VolumeServerState = 'unchecked' | 'ok' | 'unreachable';

type StatusDisplay = { status: PresentationStatus; label?: string; };
type StepDisplay = StatusDisplay & { name: string; };

const STEP_STATUS: Record<string, PresentationStatus> = {
    pending: 'waiting',
    in_progress: 'running',
    completed: 'complete',
    warning: 'warning',
    error: 'failed',
};

function presentStep(step: BackendStep): StepDisplay {
    const status = STEP_STATUS[step.status] ?? 'unknown';
    if (step.name === 'Building Docker Image') {
        // The Rust endpoint sets this when its service is running; it does not
        // observe an image build. Dev Compose/container rows have distinct names.
        return { name: 'Backend service', status };
    }
    if (step.name === 'Connecting to Volume Server') {
        return {
            name: 'Volume server (backend)',
            status,
            label: status === 'complete' ? 'Reachable' : status === 'warning' ? 'Unreachable' : undefined,
        };
    }
    if (step.name === 'Initializing ML Models') {
        return step.status === 'warning'
            ? { name: 'ML models', status: 'waiting', label: 'Loads on first run' }
            : { name: 'ML models', status };
    }
    return { name: step.name, status };
}

export function presentInitialization(
    snapshot: BackendStatus | null,
    browserVolume?: VolumeServerState,
): { summary: StatusDisplay; steps: StepDisplay[]; } {
    if (!snapshot) {
        return { summary: { status: 'waiting', label: 'Waiting for startup status' }, steps: [] };
    }
    const steps = snapshot.initialization_steps.map(presentStep);
    if (browserVolume !== undefined) {
        steps.push({
            name: 'Volume server (browser)',
            status: browserVolume === 'ok' ? 'complete' : browserVolume === 'unreachable' ? 'warning' : 'waiting',
            label: browserVolume === 'ok' ? 'Reachable' : browserVolume === 'unreachable' ? 'Unreachable' : 'Checking',
        });
    }
    let summary: StatusDisplay;
    if (steps.some((step) => step.status === 'failed')) {
        summary = snapshot.is_ready
            ? { status: 'warning', label: 'Startup needs attention' }
            : { status: 'failed', label: 'Startup failed' };
    } else if (steps.some((step) => step.status === 'warning')) {
        summary = { status: 'warning', label: 'Startup needs attention' };
    } else if (steps.some((step) => step.status === 'running')) {
        summary = { status: 'running', label: 'Starting backend' };
    } else if (steps.some((step) => step.status === 'unknown')) {
        summary = { status: 'unknown', label: 'Startup status unknown' };
    } else if (snapshot.is_ready) {
        const containerRunning = snapshot.initialization_steps.some(
            (step) => step.name === 'Container Startup' && step.status === 'completed',
        );
        summary = { status: 'complete', label: containerRunning ? 'Container running' : 'Ready' };
    } else if (snapshot.initialization_steps.every((step) => step.status === 'completed'
        || (step.name === 'Initializing ML Models' && step.status === 'warning'))
        && snapshot.initialization_steps.some((step) => step.name === 'Initializing ML Models' && step.status === 'warning')) {
        summary = { status: 'waiting', label: 'Loads on first run' };
    } else {
        summary = { status: 'waiting', label: 'Waiting for backend' };
    }
    return { summary, steps };
}
