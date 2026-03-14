"""
SIIP Dashboard — Styles & Theme
================================
Color palette, CSS, KPI card helpers, callout formatters
"""

# Color palette — Palantir/Bloomberg-inspired
COLORS = {
    'primary': '#1B2838',      # Dark navy
    'secondary': '#2C3E50',    # Slate
    'accent': '#D4AF37',       # Gold
    'accent2': '#3498DB',      # Blue
    'success': '#2ECC71',      # Green
    'warning': '#F39C12',      # Amber
    'danger': '#E74C3C',       # Red
    'text': '#ECF0F1',         # Light text
    'muted': '#95A5A6',        # Muted text
    'bg_card': '#1E2A3A',      # Card background
    'bg_dark': '#0D1117',      # Darkest background
}

# Plotly color sequences
PLOTLY_COLORS = [
    '#3498DB', '#E74C3C', '#2ECC71', '#F39C12',
    '#9B59B6', '#1ABC9C', '#E67E22', '#D4AF37',
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'
]

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#ECF0F1', 'family': 'Inter, sans-serif'},
        'colorway': PLOTLY_COLORS,
        'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.2)'},
        'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'zerolinecolor': 'rgba(255,255,255,0.2)'},
    }
}


def inject_css():
    """Returns custom CSS string for Streamlit"""
    return """
    <style>
        /* Global */
        .stApp {
            background-color: #0D1117;
        }

        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, #1E2A3A 0%, #2C3E50 100%);
            border: 1px solid rgba(212, 175, 55, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin: 5px 0;
        }
        .kpi-value {
            font-size: 2.2em;
            font-weight: 700;
            color: #D4AF37;
            margin: 5px 0;
        }
        .kpi-label {
            font-size: 0.85em;
            color: #95A5A6;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .kpi-delta {
            font-size: 0.8em;
            color: #2ECC71;
        }

        /* Evidence panels */
        .evidence-panel {
            background: #1E2A3A;
            border: 1px solid rgba(52, 152, 219, 0.3);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .evidence-label {
            font-size: 0.75em;
            color: #3498DB;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
            font-weight: 600;
        }

        /* Section headers */
        .section-header {
            border-bottom: 2px solid #D4AF37;
            padding-bottom: 8px;
            margin-bottom: 15px;
            font-size: 1.1em;
            color: #ECF0F1;
        }

        /* Metric highlight */
        .metric-highlight {
            background: rgba(212, 175, 55, 0.1);
            border-left: 3px solid #D4AF37;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }

        /* Verdict badges */
        .verdict-invest {
            background: rgba(46, 204, 113, 0.2);
            border: 2px solid #2ECC71;
            color: #2ECC71;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
        }
        .verdict-monitor {
            background: rgba(243, 156, 18, 0.2);
            border: 2px solid #F39C12;
            color: #F39C12;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
        }
        .verdict-avoid {
            background: rgba(231, 76, 60, 0.2);
            border: 2px solid #E74C3C;
            color: #E74C3C;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
        }

        /* Strategy cards */
        .strategy-card {
            background: #1E2A3A;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 8px 0;
        }
        .strategy-card h4 {
            color: #D4AF37;
            margin: 0 0 8px 0;
        }

        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #1B2838;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E2A3A;
            border-radius: 8px;
            color: #95A5A6;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2C3E50;
            color: #D4AF37;
        }
    </style>
    """


def kpi_card(label, value, delta=None):
    """Generate HTML for a KPI metric card"""
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def evidence_panel(label, content=""):
    """Generate HTML for an evidence panel header"""
    return f"""
    <div class="evidence-panel">
        <div class="evidence-label">{label}</div>
        {content}
    </div>
    """


def section_header(title):
    """Generate HTML for a section header"""
    return f'<div class="section-header">{title}</div>'


def metric_highlight(text):
    """Generate HTML for a metric highlight bar"""
    return f'<div class="metric-highlight">{text}</div>'


def verdict_badge(verdict):
    """Generate HTML for a verdict badge"""
    css_class = f'verdict-{verdict.lower()}'
    return f'<div class="{css_class}">{verdict.upper()}</div>'


def format_currency(value):
    """Format large numbers as currency"""
    if abs(value) >= 1e12:
        return f"${value/1e12:.1f}T"
    elif abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.0f}K"
    else:
        return f"${value:.0f}"


def format_pct(value):
    """Format as percentage"""
    return f"{value:.1%}"
