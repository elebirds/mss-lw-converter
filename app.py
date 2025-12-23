import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import io
import convert

st.set_page_config(layout="wide", page_title="FEDO Data Converter")

st.title("Time-Energy ↔ L-Omega Interactive Map")

# --- Sidebar ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload FEDO Data File (.txt)", type=["txt"])
st.sidebar.info("💡 **Tip:** Use 'Lasso Select' on the left plot to draw a slanted box (OBB).")

@st.cache_data
def load_data(file_obj):
    try:
        if file_obj is None:
            # Use the default file in the current directory
            file_path = "MSS1A_20240524T045957_20240524T054530_FEDO_v1.1.txt"
            try:
                df, e_cen, e_cols = convert.load_fedo_data(file_path)
            except FileNotFoundError:
                return None
        else:
            # Read uploaded file
            stringio = io.StringIO(file_obj.getvalue().decode("utf-8", errors='ignore'))
            lines = stringio.readlines()
            df, e_cen, e_cols = convert.load_fedo_data_from_lines(lines)
            
        data = convert.calculate_physics_data(df, e_cen, e_cols)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data(uploaded_file)

if data is None:
    st.warning("Please upload a data file or ensure the default file exists.")
    st.stop()

# Extract data
Time = data['time'] # (N,)
L = data['L']       # (N,)
E = data['E']       # (M,)
Flux = data['Flux'] # (N, M)
Wd = data['Wd']     # (N, M)

N = len(Time)
M = len(E)

# --- Prepare Data for L-Omega Plot (Scatter) ---
L_flat = np.repeat(L, M)
Time_flat = np.repeat(Time, M)
E_flat = np.tile(E, N)
Wd_flat = Wd.flatten()
Flux_flat = Flux.flatten()

# Filter data for plotting
with np.errstate(divide='ignore', invalid='ignore'):
    Flux_log = np.log10(Flux_flat)

mask = np.isfinite(Flux_log) & (Flux_flat > 0)

# Apply mask
L_plot = L_flat[mask]
Wd_plot = Wd_flat[mask]
Flux_plot = Flux_log[mask]
Time_plot = Time_flat[mask]
E_plot = E_flat[mask]

# --- Layout ---
col_left, col_right = st.columns(2)

# --- Retrieve Selections from Session State ---
selection_right = st.session_state.get("right_plot")
selection_left = st.session_state.get("left_plot")

# --- Create Base Figures ---

# Right Figure (L vs Omega)
fig_right = go.Figure(data=go.Scattergl(
    x=L_plot,
    y=Wd_plot,
    mode='markers',
    marker=dict(
        size=3,
        color=Flux_plot,
        colorscale='Jet',
        showscale=True,
        colorbar=dict(title='log10(Flux)', x=1.1)
    ),
    text=[f"L: {l:.2f}<br>Wd: {w:.2f}<br>Time: {t}<br>E: {e:.1f}" 
          for l, w, t, e in zip(L_plot, Wd_plot, Time_plot, E_plot)],
    hoverinfo='text'
))
fig_right.update_layout(
    xaxis_title="L-shell",
    yaxis_title="Drift Frequency (rad/h)",
    height=600,
    dragmode='lasso'
)
w_cor = 2 * np.pi / 24.0
fig_right.add_hline(y=w_cor, line_dash="dash", line_color="white", annotation_text="Earth Corotation")

# Left Figure (Time vs Energy)
with np.errstate(divide='ignore', invalid='ignore'):
    Flux_matrix_log = np.log10(Flux.T)
    
fig_left = go.Figure()

# 1. Heatmap (Background)
fig_left.add_trace(go.Heatmap(
    x=Time,
    y=E,
    z=Flux_matrix_log,
    colorscale='Jet',
    colorbar=dict(title='log10(Flux)'),
    hoverinfo='skip' # Disable hover on heatmap to prioritize scatter
))

# 2. Invisible Scatter for Selection (Foreground)
# We use the same masked data as the right plot to ensure index alignment
fig_left.add_trace(go.Scattergl(
    x=Time_plot,
    y=E_plot,
    mode='markers',
    marker=dict(size=5, color='white', opacity=0), # Invisible but selectable
    hoverinfo='none',
    showlegend=False
))

fig_left.update_layout(
    xaxis_title="Time",
    yaxis_title="Energy (keV)",
    yaxis_type="log",
    height=600,
    dragmode='lasso' # Default to lasso for OBB-like selection
)

# --- Apply Highlights ---

# 1. Right -> Left (Highlight points on Heatmap)
if selection_right and selection_right.get("selection") and selection_right["selection"].get("points"):
    indices = [p["point_index"] for p in selection_right["selection"]["points"]]
    if indices:
        sel_times = Time_plot[indices]
        sel_energies = E_plot[indices]
        fig_left.add_trace(go.Scatter(
            x=sel_times,
            y=sel_energies,
            mode='markers',
            marker=dict(color='white', size=6, symbol='circle-open', line=dict(width=2)),
            name='Selected',
            showlegend=False
        ))
        st.toast(f"Highlighted {len(indices)} points on Left Plot")

# 2. Left -> Right (Highlight points on Scatter)
mask_highlight = np.zeros(len(Time_plot), dtype=bool)
has_left_selection = False

if selection_left and selection_left.get("selection"):
    # Handle Point Selection (Lasso/Points from the invisible scatter layer)
    # Since we used the exact same data arrays (Time_plot, E_plot) for the invisible scatter
    # as we did for the Right Plot, the point_indices are identical!
    points = selection_left["selection"].get("points")
    if points:
        has_left_selection = True
        indices = [p["point_index"] for p in points]
        mask_highlight[indices] = True

    # Handle Box Selection (Fallback if user switches tool)
    box = selection_left["selection"].get("box")
    if box:
        has_left_selection = True
        for b in box:
            x_min = pd.to_datetime(b['x'][0])
            x_max = pd.to_datetime(b['x'][1])
            y_min = b['y'][0]
            y_max = b['y'][1]
            
            t_mask = (Time_plot >= x_min) & (Time_plot <= x_max)
            e_mask = (E_plot >= y_min) & (E_plot <= y_max)
            mask_highlight |= (t_mask & e_mask)

if has_left_selection and np.any(mask_highlight):
    h_L = L_plot[mask_highlight]
    h_Wd = Wd_plot[mask_highlight]
    fig_right.add_trace(go.Scattergl(
        x=h_L,
        y=h_Wd,
        mode='markers',
        marker=dict(color='white', size=4, symbol='circle-open', line=dict(width=1)),
        name='Selected',
        showlegend=False
    ))
    st.toast(f"Highlighted {np.sum(mask_highlight)} points on Right Plot")

# --- Render ---
with col_left:
    st.subheader("Time vs Energy (Observation)")
    # Removed caption to align with right plot
    st.plotly_chart(fig_left, on_select="rerun", selection_mode=["lasso", "box"], key="left_plot", width="stretch")

with col_right:
    st.subheader("L vs Drift Frequency")
    st.plotly_chart(fig_right, on_select="rerun", selection_mode=["lasso", "box", "points"], key="right_plot", width="stretch")
