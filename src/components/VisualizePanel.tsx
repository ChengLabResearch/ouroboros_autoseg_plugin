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
            borderRadius:'8px', 
            color:'#666', 
            border: '1px solid #333'
        }}>
            {children || "Visualization Area"}
        </div>
    );
}