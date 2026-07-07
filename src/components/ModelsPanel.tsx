import { useEffect, useState } from 'react';
import styles from '../assets/styles.module.css';
import { BACKEND_URL } from '../config';

type DownloadState = 'idle' | 'downloading' | 'downloaded' | 'error';

export default function ModelsPanel({ connected }: { connected: boolean }) {
    const [token, setToken] = useState('');
    const [officialSam3State, setOfficialSam3State] = useState<DownloadState>('idle');
    const [medicalSam3State, setMedicalSam3State] = useState<DownloadState>('idle');
    const [officialSam3Error, setOfficialSam3Error] = useState<string | null>(null);
    const [medicalSam3Error, setMedicalSam3Error] = useState<string | null>(null);

    const refreshModelStatuses = async () => {
        try {
            const res = await fetch(`${BACKEND_URL}/model-status`);
            if (!res.ok) return;
            const data = await res.json();
            setOfficialSam3State(data?.models?.sam3 ? 'downloaded' : 'idle');
            setMedicalSam3State(data?.models?.medical_sam3 ? 'downloaded' : 'idle');
        } catch (e) {
            console.error('Failed to fetch model status:', e);
        }
    };

    useEffect(() => {
        if (!connected) {
            setOfficialSam3State('idle');
            setMedicalSam3State('idle');
            return;
        }
        refreshModelStatuses();
        const interval = setInterval(refreshModelStatuses, 5000);
        return () => clearInterval(interval);
    }, [connected]);

    const downloadModel = async (
        modelType: string,
        hfToken: string | undefined,
        setState: (s: DownloadState) => void,
        setError: (msg: string | null) => void
    ) => {
        setState('downloading');
        setError(null);
        try {
            const res = await fetch(`${BACKEND_URL}/download-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType, hf_token: hfToken ?? '' })
            });
            if (res.ok) {
                setState('downloaded');
                await refreshModelStatuses();
            } else {
                const detail = await res.text().catch(() => '');
                const message = detail
                    ? `${res.status}: ${detail.slice(0, 200)}`
                    : `Backend rejected the download request (HTTP ${res.status}).`;
                setError(message);
                setState('error');
            }
        } catch (e) {
            const message = e instanceof Error ? e.message : String(e);
            setError(message);
            setState('error');
        }
    };

    const buttonLabel = (state: DownloadState): string => {
        if (state === 'downloading') return 'Downloading...';
        if (state === 'downloaded') return 'Downloaded';
        if (state === 'error') return 'Retry';
        return 'Download';
    };

    const buttonClass = (state: DownloadState): string => {
        if (state === 'downloaded') return `${styles.downloadBtn} ${styles.downloadBtnSuccess}`;
        if (state === 'error') return `${styles.downloadBtn} ${styles.downloadBtnError}`;
        return styles.downloadBtn;
    };

    return (
        <div className={styles.container}>
            <div className={styles.headerRow}>
                <span className={styles.headerTitle}>MODELS</span>
            </div>
            <div className={styles.scrollContent}>
                <div className={styles.section}>
                    <div className={styles.sam3Container}>
                        <div className={styles.row}>
                            <span className={styles.label}>SAM3 (Official; Authenticated)</span>
                        </div>
                        <div className={styles.row}>
                            <input
                                className={styles.tokenInput}
                                type="password"
                                value={token}
                                onChange={e => setToken(e.target.value)}
                                placeholder="Paste HF Token"
                            />
                            <button
                                className={buttonClass(officialSam3State)}
                                onClick={() => downloadModel('sam3', token, setOfficialSam3State, setOfficialSam3Error)}
                                disabled={
                                    officialSam3State === 'downloading' ||
                                    officialSam3State === 'downloaded' ||
                                    !token
                                }
                            >
                                {buttonLabel(officialSam3State)}
                            </button>
                        </div>
                        {officialSam3Error && (
                            <div className={styles.inlineError} role="alert">{officialSam3Error}</div>
                        )}
                    </div>

                    <div className={styles.sam3Container}>
                        <div className={styles.row}>
                            <span className={styles.label}>SAM3 (Medical)</span>
                            <button
                                className={buttonClass(medicalSam3State)}
                                onClick={() => downloadModel('medical_sam3', undefined, setMedicalSam3State, setMedicalSam3Error)}
                                disabled={
                                    medicalSam3State === 'downloading' ||
                                    medicalSam3State === 'downloaded'
                                }
                            >
                                {buttonLabel(medicalSam3State)}
                            </button>
                        </div>
                        {medicalSam3Error && (
                            <div className={styles.inlineError} role="alert">{medicalSam3Error}</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
