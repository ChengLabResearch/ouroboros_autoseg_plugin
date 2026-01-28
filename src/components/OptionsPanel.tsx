import React, { useState } from 'react';
import styles from '../assets/styles.module.css';

type Props = { 
    onSubmit: (d: any) => void; 
    isRunning: boolean; 
};

const BACKEND_URL = "http://localhost:8686";
export default function OptionsPanel({ onSubmit, isRunning }: Props) {
    const [fp, setFp] = useState('');
    const [outFp, setOutFp] = useState('');
    const [model, setModel] = useState('sam2_hiera_base_plus');
    const [predictor, setPredictor] = useState('ImagePredictor');
    const [token, setToken] = useState('');

    // Download States
    const [sam2Status, setSam2Status] = useState('Download');
    const [sam3Status, setSam3Status] = useState('Download');

    const handleDrop = (e: React.DragEvent, setFn: (s: string) => void) => {
        e.preventDefault();
        e.stopPropagation();
        const data = e.dataTransfer.getData("text/plain");
        if (data) setFn(data);
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const downloadModel = async (modelType: string, hfToken: string, setStatus: (s: string) => void) => {
        setStatus('Downloading...');
        try {
            const res = await fetch(`${BACKEND_URL}/download-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_type: modelType, hf_token: hfToken })
            });
            if (res.ok) setStatus('Done');
            else {
                console.error(await res.text());
                setStatus('Error');
            }
        } catch (e) {
            console.error(e);
            setStatus('Error');
        }
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
                            className={styles.downloadBtn} 
                            onClick={() => downloadModel('sam2_hiera_base_plus', '', setSam2Status)}
                            disabled={sam2Status !== 'Download'}
                        >
                            {sam2Status}
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
                                className={styles.downloadBtn} 
                                onClick={() => downloadModel('sam3', token, setSam3Status)}
                                disabled={sam3Status !== 'Download' || !token}
                            >
                                {sam3Status}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}