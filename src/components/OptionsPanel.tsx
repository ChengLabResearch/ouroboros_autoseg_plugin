import React, { useEffect, useState } from 'react';
import styles from '../assets/styles.module.css';
import { BACKEND_URL } from '../config';

type Props = { 
    onSubmit: (d: any) => void; 
    isRunning: boolean; 
    connected: boolean;
};

export default function OptionsPanel({ onSubmit, isRunning, connected }: Props) {
    const [fp, setFp] = useState('');
    const [outFp, setOutFp] = useState('');
    const [model, setModel] = useState('sam2_hiera_base_plus');
    const [predictor, setPredictor] = useState('ImagePredictor');
    const [token, setToken] = useState('');

    type DownloadState = 'idle' | 'downloading' | 'downloaded' | 'error';
    const [sam2State, setSam2State] = useState<DownloadState>('idle');
    const [sam3State, setSam3State] = useState<DownloadState>('idle');

    const normalizeFileUri = (uri: string): string => {
        let cleaned = uri.trim();
        if (cleaned.startsWith('file://')) {
            cleaned = cleaned.replace(/^file:\/\//, '');
            cleaned = decodeURIComponent(cleaned);
            if (/^\/[A-Za-z]:\//.test(cleaned)) {
                cleaned = cleaned.slice(1);
            }
        }
        return cleaned.trim();
    };

    const parseTextToPath = (text: string): string | null => {
        const trimmed = text.trim();
        if (!trimmed) return null;
        if (trimmed.startsWith('file://')) {
            return normalizeFileUri(trimmed);
        }
        if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            try {
                const data = JSON.parse(trimmed);
                if (typeof data === 'string') return data;
                if (Array.isArray(data)) {
                    const first = data.find((item) => typeof item === 'string');
                    return first ?? null;
                }
                if (data && typeof data === 'object') {
                    const keys = ['path', 'filePath', 'filepath', 'file_path', 'uri', 'url'];
                    for (const key of keys) {
                        const val = (data as any)[key];
                        if (typeof val === 'string') return val;
                    }
                }
            } catch {}
        }
        return trimmed;
    };

    const extractPathFromDataTransfer = (dt: DataTransfer): string | null => {
        if (dt.items && dt.items.length > 0) {
            for (const item of Array.from(dt.items)) {
                if (item.kind === 'file') {
                    const file = item.getAsFile();
                    if (file) {
                        const anyFile = file as any;
                        return (typeof anyFile.path === 'string' && anyFile.path) || file.name || null;
                    }
                }
            }
        }

        if (dt.files && dt.files.length > 0) {
            const file = dt.files[0];
            const anyFile = file as any;
            return (typeof anyFile.path === 'string' && anyFile.path) || file.name || null;
        }

        const types = Array.from(dt.types || []);
        const preferredTypes = [
            'text/uri-list',
            'text/plain',
            'application/json',
            'text/json',
            'application/x-ouroboros-path',
            'application/x-ouroboros-file',
            'application/x-file-path'
        ];

        const orderedTypes = [
            ...preferredTypes.filter((t) => types.includes(t)),
            ...types.filter((t) => !preferredTypes.includes(t))
        ];

        for (const type of orderedTypes) {
            const raw = dt.getData(type);
            if (!raw) continue;
            if (type === 'text/uri-list') {
                const line = raw
                    .split(/\r?\n/)
                    .map((entry) => entry.trim())
                    .find((entry) => entry && !entry.startsWith('#'));
                if (line) return normalizeFileUri(line);
                continue;
            }
            const parsed = parseTextToPath(raw);
            if (parsed) return parsed;
        }

        return null;
    };

    const handleDrop = (e: React.DragEvent, setFn: (s: string) => void) => {
        e.preventDefault();
        e.stopPropagation();
        const data = extractPathFromDataTransfer(e.dataTransfer);
        if (data) setFn(data);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer) {
            e.dataTransfer.dropEffect = 'copy';
        }
    };

    const refreshModelStatuses = async () => {
        if (!connected) return;
        try {
            const res = await fetch(`${BACKEND_URL}/model-status`);
            if (!res.ok) {
                return;
            }
            const data = await res.json();
            setSam2State(data?.models?.sam2_hiera_base_plus ? 'downloaded' : 'idle');
            setSam3State(data?.models?.sam3 ? 'downloaded' : 'idle');
        } catch (e) {
            console.error('Failed to fetch model status:', e);
        }
    };

    useEffect(() => {
        if (!connected) {
            setSam2State('idle');
            setSam3State('idle');
            return;
        }
        refreshModelStatuses();
        const interval = setInterval(refreshModelStatuses, 5000);
        return () => clearInterval(interval);
    }, [connected]);

    const downloadModel = async (
        modelType: string,
        hfToken: string,
        setState: (s: DownloadState) => void
    ) => {
        setState('downloading');
        try {
            const res = await fetch(`${BACKEND_URL}/download-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType, hf_token: hfToken })
            });
            if (res.ok) {
                await refreshModelStatuses();
            } else {
                console.error(await res.text());
                setState('error');
            }
        } catch (e) {
            console.error(e);
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

    const canRun = !isRunning && fp && outFp;

    return (
        <div className={styles.container}>
            <div className={styles.scrollContent}>
                
                {/* --- OPTIONS SECTION --- */}
                <div className={styles.headerRow}>
                    <span className={styles.headerTitle}>OPTIONS</span>
                    <button 
                        className={styles.playBtn} 
                        onClick={() => onSubmit({filePath: fp, outputFile: outFp, modelType: model, predictor_type: predictor})} 
                        disabled={!canRun}
                        title="Run Segmentation"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                    </button>
                </div>

                <div className={styles.section}>
                    <div className={styles.row}>
                        <span className={styles.label}>Input File</span>
                        <div className={styles.inputContainer}>
                            <input 
                                className={styles.input} 
                                value={fp} 
                                onChange={e => setFp(e.target.value)} 
                                placeholder="Drag file here..." 
                                onDrop={(e) => handleDrop(e, setFp)} 
                                onDragOver={handleDragOver} 
                            />
                        </div>
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Output File</span>
                        <div className={styles.inputContainer}>
                            <input 
                                className={styles.input} 
                                value={outFp} 
                                onChange={e => setOutFp(e.target.value)} 
                                placeholder="/path/to/output.tif" 
                                onDrop={(e) => handleDrop(e, setOutFp)} 
                                onDragOver={handleDragOver} 
                            />
                        </div>
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Predictor</span>
                        <div className={styles.inputContainer}>
                            <select className={styles.select} value={predictor} onChange={e => setPredictor(e.target.value)}>
                                <option value="ImagePredictor">Image Predictor</option>
                                <option value="VideoPredictor">Video Predictor</option>
                            </select>
                        </div>
                    </div>

                    <div className={styles.row}>
                        <span className={styles.label}>Model</span>
                        <div className={styles.inputContainer}>
                            <select className={styles.select} value={model} onChange={e => setModel(e.target.value)}>
                                <optgroup label="SAM 2">
                                    <option value="sam2_hiera_base_plus">SAM2 Base+</option>
                                    <option value="sam2_hiera_large">SAM2 Large</option>
                                </optgroup>
                                <optgroup label="SAM 3">
                                    <option value="sam3">SAM3</option>
                                </optgroup>
                            </select>
                        </div>
                    </div>
                </div>

                {/* --- MODELS SECTION --- */}
                <div className={styles.headerRow}>
                    <span className={styles.headerTitle}>MODELS</span>
                </div>

                <div className={styles.section}>
                    {/* SAM 2 */}
                    <div className={styles.row}>
                        <span className={styles.label}>SAM 2 (Official)</span>
                        <button 
                            className={buttonClass(sam2State)}
                            onClick={() => downloadModel('sam2_hiera_base_plus', '', setSam2State)}
                            disabled={sam2State === 'downloading' || sam2State === 'downloaded'}
                        >
                            {buttonLabel(sam2State)}
                        </button>
                    </div>

                    {/* SAM 3 */}
                    <div className={styles.sam3Container}>
                        <div className={styles.row}>
                            <span className={styles.label}>SAM 3 (Authenticated)</span>
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
                                className={buttonClass(sam3State)}
                                onClick={() => downloadModel('sam3', token, setSam3State)}
                                disabled={
                                    sam3State === 'downloading' ||
                                    sam3State === 'downloaded' ||
                                    !token
                                }
                            >
                                {buttonLabel(sam3State)}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
