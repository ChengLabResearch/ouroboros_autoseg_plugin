import React from 'react';
import styles from '../assets/styles.module.css';

export default function VisualizePanel({ children }: { children?: React.ReactNode }) {
    return (
        <div className={styles.vizPlaceholder}>
            {children || "Visualization Area"}
        </div>
    );
}