import React from 'react';

export default function VisualizePanel({ children }: { children?: React.ReactNode }) {
    return (
        <div style={{
            backgroundColor: 'var(--panel-background)', 
            width:'100%', 
            height:'100%', 
            display:'flex', 
            alignItems:'center', 
            justifyContent:'center', 
            color:'#666', 
        }}>
            {children || "Visualization Area"}
        </div>
    );
}