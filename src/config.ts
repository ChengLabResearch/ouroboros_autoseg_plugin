export const BACKEND_URL =
    import.meta.env.VITE_BACKEND_URL ??
    (import.meta.env.DEV ? '/api' : 'http://localhost:8686');
