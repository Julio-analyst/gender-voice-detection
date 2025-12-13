"""
Admin Panel - Gradio Interface
Dashboard untuk monitoring dan management model
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from datetime import datetime
import json

# Handle imports
try:
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config

# Initialize config
config = get_config()


def authenticate(username, password):
    """
    Authenticate admin user
    
    Args:
        username: Admin username
        password: Admin password
        
    Returns:
        True if authenticated, False otherwise
    """
    admin_username = os.getenv('ADMIN_USERNAME', 'admin')
    admin_password = os.getenv('ADMIN_PASSWORD', 'mlops2024!')
    
    return username == admin_username and password == admin_password


def get_feedback_stats():
    """Get feedback statistics"""
    try:
        feedback_file = Path('data/feedback/feedback.csv')
        
        if not feedback_file.exists():
            return {
                'total': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        df = pd.read_csv(feedback_file)
        
        return {
            'total': len(df),
            'correct': int(df['is_correct'].sum()),
            'incorrect': int((~df['is_correct']).sum()),
            'accuracy': float(df['is_correct'].mean()) if len(df) > 0 else 0.0
        }
    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0.0}


def view_dashboard():
    """Generate dashboard view"""
    try:
        stats = get_feedback_stats()
        threshold = int(os.getenv('FEEDBACK_THRESHOLD', 20))
        
        dashboard_md = f"""
## 📊 Dashboard Overview

### Feedback Statistics
- **Total Feedback:** {stats['total']} / {threshold}
- **Correct Predictions:** {stats['correct']} ({stats['correct']/stats['total']*100:.1f}% jika ada data)
- **Incorrect Predictions:** {stats['incorrect']}
- **Accuracy:** {stats['accuracy']:.2%}

### Auto-Retrain Status
- **Threshold:** {threshold} feedback
- **Progress:** {stats['total']}/{threshold} ({stats['total']/threshold*100:.0f}%)
- **Status:** {'🔴 Ready to retrain!' if stats['total'] >= threshold else '🟡 Collecting feedback...'}
"""
        return dashboard_md
        
    except Exception as e:
        return f"❌ Error loading dashboard: {str(e)}"


def view_feedback_table():
    """View feedback data as table"""
    try:
        feedback_file = Path('data/feedback/feedback.csv')
        
        if not feedback_file.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(feedback_file)
        
        # Sort by timestamp descending
        df = df.sort_values('timestamp', ascending=False)
        
        # Select relevant columns
        display_columns = [
            'feedback_id', 'timestamp', 'predicted_label',
            'actual_label', 'model_type', 'confidence', 'is_correct'
        ]
        
        return df[display_columns].head(50)
        
    except Exception as e:
        print(f"Error loading feedback table: {e}")
        return pd.DataFrame()


def plot_feedback_by_model():
    """Plot feedback accuracy by model type"""
    try:
        feedback_file = Path('data/feedback/feedback.csv')
        
        if not feedback_file.exists():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No feedback data available',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        df = pd.read_csv(feedback_file)
        
        if len(df) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No feedback data available',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Calculate accuracy by model
        model_stats = df.groupby('model_type').agg({
            'is_correct': ['sum', 'count', 'mean']
        }).reset_index()
        
        model_stats.columns = ['model_type', 'correct', 'total', 'accuracy']
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot - Total feedback by model
        ax1.bar(model_stats['model_type'], model_stats['total'], color=['#667eea', '#764ba2', '#f093fb'])
        ax1.set_xlabel('Model Type', fontsize=12)
        ax1.set_ylabel('Total Feedback', fontsize=12)
        ax1.set_title('Feedback Count by Model', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Bar plot - Accuracy by model
        ax2.bar(model_stats['model_type'], model_stats['accuracy'], color=['#667eea', '#764ba2', '#f093fb'])
        ax2.set_xlabel('Model Type', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy by Model', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(model_stats.iterrows()):
            ax1.text(i, row['total'], f"{int(row['total'])}", 
                    ha='center', va='bottom', fontsize=10)
            ax2.text(i, row['accuracy'], f"{row['accuracy']:.2%}", 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error plotting feedback: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def plot_feedback_timeline():
    """Plot feedback submission over time"""
    try:
        feedback_file = Path('data/feedback/feedback.csv')
        
        if not feedback_file.exists():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No feedback data available',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        df = pd.read_csv(feedback_file)
        
        if len(df) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No feedback data available',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Cumulative count
        df['cumulative_count'] = range(1, len(df) + 1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df['timestamp'], df['cumulative_count'], 
               marker='o', linestyle='-', linewidth=2, markersize=4,
               color='#667eea')
        
        # Add threshold line
        threshold = int(os.getenv('FEEDBACK_THRESHOLD', 20))
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold})')
        
        ax.set_xlabel('Waktu', fontsize=12)
        ax.set_ylabel('Jumlah Feedback Kumulatif', fontsize=12)
        ax.set_title('Timeline Feedback Collection', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error plotting timeline: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def trigger_retrain(retrain_model_type):
    """Trigger manual model retraining"""
    try:
        # Check if enough feedback
        stats = get_feedback_stats()
        
        if stats['total'] == 0:
            return "❌ Tidak ada feedback untuk training. Minimal 1 feedback diperlukan."
        
        return f"""
⏳ **Retraining dimulai untuk model {retrain_model_type}...**

Proses ini akan:
1. Load feedback data ({stats['total']} samples)
2. Prepare dataset baru
3. Retrain model {retrain_model_type}
4. Evaluate dan compare dengan model lama
5. Deploy jika performa lebih baik

**Note:** Ini adalah simulasi. Implementasi lengkap akan menggunakan background task (Celery).

Untuk trigger retraining sebenarnya, jalankan:
```bash
python src/training/auto_retrain.py --model {retrain_model_type}
```
"""
        
    except Exception as e:
        return f"❌ Error triggering retrain: {str(e)}"


def export_feedback_report():
    """Export feedback report"""
    try:
        feedback_file = Path('data/feedback/feedback.csv')
        
        if not feedback_file.exists():
            return "❌ Tidak ada data feedback untuk di-export"
        
        # Copy to reports directory
        reports_dir = Path('reports/feedback')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = reports_dir / f'feedback_export_{timestamp}.csv'
        
        import shutil
        shutil.copy(feedback_file, export_path)
        
        # Also create summary JSON
        stats = get_feedback_stats()
        summary_path = reports_dir / f'feedback_summary_{timestamp}.json'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return f"""
✅ **Feedback report berhasil di-export!**

**File yang dibuat:**
- CSV: `{export_path}`
- Summary: `{summary_path}`

**Total feedback:** {stats['total']}
**Accuracy:** {stats['accuracy']:.2%}
"""
        
    except Exception as e:
        return f"❌ Error exporting report: {str(e)}"


def create_admin_panel():
    """Create admin panel interface"""
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Admin Panel - Gender Voice Detection") as demo:
        
        gr.Markdown(
            """
            # 🔐 Admin Panel
            ### Gender Voice Detection - MLOps Management Dashboard
            """
        )
        
        with gr.Tabs():
            # Tab 1: Dashboard
            with gr.Tab("📊 Dashboard"):
                gr.Markdown("## Overview Sistem")
                
                refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                dashboard_md = gr.Markdown()
                
                refresh_btn.click(fn=view_dashboard, outputs=dashboard_md)
                
                gr.Markdown("---")
                
                gr.Markdown("## 📈 Visualisasi Feedback")
                
                with gr.Row():
                    feedback_plot = gr.Plot(label="Feedback by Model")
                    timeline_plot = gr.Plot(label="Feedback Timeline")
                
                plot_btn = gr.Button("📊 Generate Plots")
                plot_btn.click(
                    fn=lambda: (plot_feedback_by_model(), plot_feedback_timeline()),
                    outputs=[feedback_plot, timeline_plot]
                )
            
            # Tab 2: Feedback Data
            with gr.Tab("💬 Feedback Data"):
                gr.Markdown("## Data Feedback Pengguna")
                
                feedback_table = gr.Dataframe(
                    headers=[
                        'feedback_id', 'timestamp', 'predicted_label',
                        'actual_label', 'model_type', 'confidence', 'is_correct'
                    ],
                    label="Recent Feedback (50 terakhir)"
                )
                
                load_table_btn = gr.Button("📥 Load Feedback Table")
                load_table_btn.click(fn=view_feedback_table, outputs=feedback_table)
                
                gr.Markdown("---")
                
                export_btn = gr.Button("📤 Export Feedback Report", variant="secondary")
                export_status = gr.Markdown()
                
                export_btn.click(fn=export_feedback_report, outputs=export_status)
            
            # Tab 3: Model Management
            with gr.Tab("🤖 Model Management"):
                gr.Markdown("## Retrain Model")
                
                gr.Markdown(
                    """
                    Trigger manual retraining model dengan feedback data yang sudah terkumpul.
                    """
                )
                
                retrain_model_choice = gr.Radio(
                    choices=["LSTM", "RNN", "GRU"],
                    value="LSTM",
                    label="Pilih Model untuk Retrain"
                )
                
                retrain_btn = gr.Button("🚀 Trigger Retrain", variant="primary", size="lg")
                retrain_status = gr.Markdown()
                
                retrain_btn.click(
                    fn=trigger_retrain,
                    inputs=retrain_model_choice,
                    outputs=retrain_status
                )
                
                gr.Markdown("---")
                
                gr.Markdown(
                    """
                    ## ℹ️ Auto-Retrain Configuration
                    
                    **Threshold:** 20 feedback (dapat diubah di `.env`)
                    
                    **Status:** Enabled (AUTO_RETRAIN_ENABLED=true)
                    
                    Model akan otomatis di-retrain ketika feedback mencapai threshold.
                    """
                )
            
            # Tab 4: System Info
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown(
                    """
                    ## System Information
                    
                    ### Models Location
                    - LSTM: `models/lstm_production.h5`
                    - RNN: `models/rnn_production.h5`
                    - GRU: `models/gru_production.h5`
                    
                    ### Data Paths
                    - Raw Audio: `data/raw/`
                    - Processed: `data/processed/`
                    - MFCC: `data/mfcc/`
                    - Feedback: `data/feedback/`
                    
                    ### MLflow Tracking
                    - URI: https://dagshub.com/Julio-analyst/gender-voice-detection.mlflow
                    - Experiment: gender-voice-detection
                    
                    ### Environment
                    - Python: 3.11.7
                    - TensorFlow: 2.13.0
                    - Gradio: 4.7.1
                    
                    ### Configuration
                    Edit `.env` file untuk mengubah konfigurasi sistem.
                    """
                )
        
        # Load dashboard on startup
        demo.load(fn=view_dashboard, outputs=dashboard_md)
    
    return demo


def main():
    """Main function to launch admin panel"""
    print("="*80)
    print("Admin Panel - Gender Voice Detection")
    print("="*80)
    
    # Create demo
    demo = create_admin_panel()
    
    # Launch
    port = int(os.getenv('GRADIO_SERVER_PORT', 7860)) + 1  # Use different port
    
    print(f"\n🚀 Launching Admin Panel on http://localhost:{port}")
    print(f"   Username: admin")
    print(f"   Password: {os.getenv('ADMIN_PASSWORD', 'mlops2024!')}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
