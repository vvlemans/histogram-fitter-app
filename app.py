#Student Name: Vanessa Lemanski 
#Student ID: 21212582

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats

st.set_page_config(page_title="Distribution Fitter", layout="wide")

st.title("Vanessa Lemanski's Cool Histogram and Distribution Tool")
st.write("Enter x (value) ranges and frequency pairs (e.g., 1-2, 25), or upload a CSV with x and freq columns. Only ranges are accepted (single values not supported).")

# Distributions
DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma (gamma)": stats.gamma,
    "Lognormal (lognorm)": stats.lognorm,
    "Weibull (minimum) (weibull_min)": stats.weibull_min,
    "Weibull (maximum) (weibull_max)": stats.weibull_max,
    "Exponential (expon)": stats.expon,
    "Uniform (uniform)": stats.uniform,
    "Beta (beta)": stats.beta,
    "Chi-square (chi2)": stats.chi2,
    "Rayleigh (rayleigh)": stats.rayleigh,
    "Logistic (logistic)": stats.logistic,
    "Cauchy (cauchy)": stats.cauchy,
}

def parse_range(s):
    """Parse x value - must be a range like '1-5' or '1 to 5'
    Returns (midpoint, low, high) or None if not a valid range"""
    s = str(s).strip()
    # Check for range patterns (1-5, 1 to 5, 1:5)
    if re.search(r'[-:]| to ', s, re.IGNORECASE):
        parts = re.split(r'[-:]| to ', s, flags=re.IGNORECASE)
        if len(parts) == 2:
            try:
                low, high = float(parts[0].strip()), float(parts[1].strip())
                if low > high:
                    low, high = high, low
                return ((low + high) / 2, low, high)  # Return midpoint and range
            except:
                pass
    # Single numbers are not accepted - return None
    return None

def parse_manual_input(text):
    """Parse manual input: x, freq pairs (one per line or comma-separated)
    Only accepts ranges (e.g., 1-2, 25). Single values are rejected.
    Returns x_vals (midpoints), freqs, and ranges (low, high) for each x"""
    x_vals, freqs, ranges = [], [], []
    lines = text.strip().split('\n')
    for line in lines:
        parts = [p.strip() for p in re.split(r'[,;\t]', line) if p.strip()]
        if len(parts) >= 2:
            range_info = parse_range(parts[0])
            try:
                freq = float(parts[1])
                # Only accept if it's a valid range (not a single number)
                if range_info is not None and freq >= 0:
                    midpoint, low, high = range_info
                    # Ensure it's actually a range (low != high)
                    if low != high:
                        x_vals.append(midpoint)
                        freqs.append(freq)
                        ranges.append((low, high))
            except:
                continue
    return np.array(x_vals), np.array(freqs), ranges

def load_csv_data(uploaded_file):
    """Load x, freq from CSV - auto-detect columns
    Returns x_vals (midpoints), freqs, and ranges"""
    try:
        df = pd.read_csv(uploaded_file)
        # Look for x/value and freq/frequency columns (case-insensitive)
        x_col = None
        freq_col = None
        for col in df.columns:
            col_lower = col.lower()
            if x_col is None and any(term in col_lower for term in ['x', 'value', 'val']):
                x_col = col
            if freq_col is None and any(term in col_lower for term in ['freq', 'frequency', 'count']):
                freq_col = col
        
        if x_col is None or freq_col is None:
            # Use first two columns (first can be text/numeric for ranges)
            if len(df.columns) >= 2:
                x_col, freq_col = df.columns[0], df.columns[1]
            else:
                return None, None, None
        
        x_vals, freqs, ranges = [], [], []
        for idx in df.index:
            range_info = parse_range(str(df.loc[idx, x_col]))
            freq_val = pd.to_numeric(df.loc[idx, freq_col], errors='coerce')
            if range_info is not None and not pd.isna(freq_val) and freq_val >= 0:
                midpoint, low, high = range_info
                # Only accept ranges (low != high), reject single values
                if low != high:
                    x_vals.append(midpoint)
                    freqs.append(float(freq_val))
                    ranges.append((low, high))
        
        return np.array(x_vals), np.array(freqs), ranges
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None, None, None

def split_params(params):
    """Split params into shape, loc, scale"""
    params_list = list(params)
    if len(params_list) < 2:
        return [], 0.0, 1.0
    return params_list[:-2], params_list[-2], params_list[-1]

def compute_metrics(x_vals, freqs, dist_obj, params):
    """Compute MAE and max error"""
    shape_params, loc, scale = split_params(params)
    dist_frozen = dist_obj(*shape_params, loc=loc, scale=scale)
    
    # Normalize frequencies to density
    total = np.sum(freqs)
    if total == 0:
        return np.nan, np.nan
    density = freqs / total
    
    # Get PDF values
    try:
        pdf_vals = dist_frozen.pdf(x_vals)
        pdf_vals = np.nan_to_num(pdf_vals, nan=0.0)
        # Normalize PDF
        pdf_norm = pdf_vals / np.sum(pdf_vals) if np.sum(pdf_vals) > 0 else pdf_vals
    except:
        return np.nan, np.nan
    
    errors = np.abs(density - pdf_norm)
    return np.mean(errors), np.max(errors)

# Sidebar
st.sidebar.header("Controls")
data_source = st.sidebar.radio("Data Source", ["Manual Entry", "CSV Upload"])
selected_dist = st.sidebar.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
fitting_mode = st.sidebar.radio("Fitting Mode", ["Automatic", "Manual"])

# Data Input
st.header("Data Input")
x_vals, freqs, ranges = None, None, None

if data_source == "Manual Entry":
    manual_text = st.text_area(
        "Enter x, freq pairs (one per line, comma-separated). Use ranges only (e.g., 1-2, 25):",
        height=200,
        placeholder="Example:\n1-2, 25\n2-3, 30\n3-4, 5\n\nNote: Only ranges are accepted (e.g., 1-2, not just 1)"
    )
    if manual_text:
        x_vals, freqs, ranges = parse_manual_input(manual_text)

else:
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        x_vals, freqs, ranges = load_csv_data(uploaded_file)
        if x_vals is not None:
            st.success(f"Loaded {len(x_vals)} data points")

if x_vals is None or freqs is None:
    st.warning("No valid data. Please enter x, freq pairs using ranges (e.g., 1-2, 25) or upload CSV.")
    st.stop()

if len(x_vals) == 0 or len(freqs) == 0:
    st.warning("No valid data. Please enter x, freq pairs using ranges (e.g., 1-2, 25) or upload CSV. Single values are not accepted.")
    st.stop()

# Ensure we have ranges (required)
if not ranges or len(ranges) == 0:
    st.warning("Ranges are required. Please use range format (e.g., 1-2, 25) instead of single values.")
    st.stop()

# Ensure ranges is a list (empty if None)
if ranges is None:
    ranges = []

# Ensure x_vals and freqs are not None and have the same length
if x_vals is None or freqs is None:
    st.error("Data loading error: x_vals or freqs is None")
    st.stop()

if len(x_vals) != len(freqs):
    min_len = min(len(x_vals), len(freqs))
    x_vals = x_vals[:min_len]
    freqs = freqs[:min_len]
    if len(ranges) > min_len:
        ranges = ranges[:min_len]

# Expand data based on frequencies for fitting (distribute across ranges)
expanded_data = []
for i, (x, f) in enumerate(zip(x_vals, freqs)):
    count = max(1, int(round(f)))
    if ranges and i < len(ranges):
        low, high = ranges[i]
        if low != high:
            # Distribute values across the range
            expanded_data.extend(np.linspace(low, high, count))
        else:
            expanded_data.extend([x] * count)
    else:
        expanded_data.extend([x] * count)
expanded_data = np.array(expanded_data)

# Fitting
dist_obj = DISTRIBUTIONS[selected_dist]
try:
    fitted_params = dist_obj.fit(expanded_data)
    fitted_params_list = list(fitted_params)
    if len(fitted_params_list) >= 2:
        fitted_params_list[-1] = max(fitted_params_list[-1], 1e-6)
    fitted_params = tuple(fitted_params_list)
except Exception as e:
    st.error(f"Could not fit distribution: {e}")
    st.stop()

shape_params_auto, loc_auto, scale_auto = split_params(fitted_params)

# Manual fitting
manual_params = None
if fitting_mode == "Manual":
    st.subheader("Manual Parameters")
    with st.expander("Adjust Parameters"):
        manual_shapes = []
        for i in range(len(shape_params_auto)):
            val = st.slider(f"Shape {i+1}", 0.1, float(shape_params_auto[i]) * 3, float(shape_params_auto[i]), key=f"s{i}")
            manual_shapes.append(val)
        
        data_range = np.max(x_vals) - np.min(x_vals)
        loc_val = st.slider("Location", float(np.min(x_vals)) - data_range, float(np.max(x_vals)) + data_range, float(loc_auto), key="loc")
        scale_val = st.slider("Scale", 1e-6, float(scale_auto) * 3, float(scale_auto), key="scale")
        manual_params = tuple(manual_shapes + [loc_val, scale_val])

active_params = manual_params if (fitting_mode == "Manual" and manual_params) else fitted_params

# Visualization
st.header("Results")

# Prepare plot data
x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
x_range = x_max - x_min if x_max > x_min else 1.0

# Expand plot range to show distribution properly
if ranges and len(ranges) == len(x_vals):
    x_min_plot = min([low for low, high in ranges])
    x_max_plot = max([high for low, high in ranges])
else:
    x_min_plot, x_max_plot = x_min, x_max

# Create x_plot with enough range to show the distribution
plot_range = x_max_plot - x_min_plot if x_max_plot > x_min_plot else 1.0
x_plot = np.linspace(x_min_plot - 0.2*plot_range, x_max_plot + 0.2*plot_range, 400)

shape_active, loc_active, scale_active = split_params(active_params)

# Ensure scale is positive (required for all distributions)
scale_active = max(float(scale_active), 1e-6)

# Create frozen distribution with proper parameter handling
try:
    if len(shape_active) > 0:
        # Ensure all shape parameters are positive where required
        shape_active = [max(float(s), 1e-6) if s <= 0 else float(s) for s in shape_active]
        dist_frozen = dist_obj(*shape_active, loc=loc_active, scale=scale_active)
    else:
        # Distributions with no shape parameters (like expon, uniform)
        dist_frozen = dist_obj(loc=loc_active, scale=scale_active)
except Exception as e1:
    # Fallback: try with all parameters as positional
    try:
        # Ensure scale in params is positive
        params_list = list(active_params)
        if len(params_list) >= 2:
            params_list[-1] = max(float(params_list[-1]), 1e-6)
        dist_frozen = dist_obj(*params_list)
    except:
        # Last resort: use loc and scale only
        dist_frozen = dist_obj(loc=loc_active, scale=scale_active)

try:
    pdf_vals = dist_frozen.pdf(x_plot)
    pdf_vals = np.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure PDF values are non-negative
    pdf_vals = np.maximum(pdf_vals, 0)
except Exception as e:
    st.warning(f"Error calculating PDF: {e}")
    pdf_vals = np.zeros_like(x_plot)

# Scale PDF to match frequency scale
total_freq = np.sum(freqs)
if total_freq == 0:
    total_freq = 1.0  # Avoid division by zero

# Calculate bin widths for proper scaling (ranges only)
bin_widths = np.array([max(high - low, 0.01) if high > low else max(x_range / max(len(x_vals), 1), 0.01) for low, high in ranges])

# Use average bin width for scaling the continuous PDF curve
avg_bin_width = np.mean(bin_widths) if len(bin_widths) > 0 else 1.0

# Scale PDF to match frequency scale accurately (ranges only)
# PDF(x) * bin_width * total_freq gives expected frequency
# Scale PDF using actual bin width for each range
pdf_scaled = np.zeros_like(pdf_vals)
for i, (low, high) in enumerate(ranges):
    bin_width = high - low if high > low else x_range / max(len(x_vals), 1)
    # Find x_plot points that fall within this exact range
    mask = (x_plot >= low) & (x_plot < high)
    # Also include the right boundary for the last point
    if i == len(ranges) - 1:
        mask = mask | (x_plot == high)
    if np.any(mask):
        # Scale PDF by this bin's actual width
        pdf_scaled[mask] = pdf_vals[mask] * bin_width * total_freq
pdf_scaled = np.maximum(pdf_scaled, 0)
# For points between ranges (gaps), scale using average bin width
uncovered = pdf_scaled == 0
if np.any(uncovered):
    pdf_scaled[uncovered] = pdf_vals[uncovered] * avg_bin_width * total_freq

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with actual frequencies (ranges only - bars touch, ranges are accurate)
# Sort by low values to ensure proper ordering
sorted_data = sorted(zip(x_vals, freqs, ranges), key=lambda t: t[2][0])
sorted_x, sorted_freqs, sorted_ranges = zip(*sorted_data) if sorted_data else ([], [], [])

# Calculate bin edges so bars touch (fill gaps between ranges)
bin_edges = []
for i, (low, high) in enumerate(sorted_ranges):
    if i == 0:
        bin_edges.append(low)  # Start at first range's low
    else:
        # Connect to previous: use midpoint if there's a gap, or exact boundary if touching
        prev_high = sorted_ranges[i-1][1]
        if prev_high < low:
            # Gap exists - use midpoint to fill it
            bin_edges.append((prev_high + low) / 2)
        else:
            # Ranges touch or overlap - use exact boundary
            bin_edges.append(low)

# Add right edge of last bar
if len(sorted_ranges) > 0:
    bin_edges.append(sorted_ranges[-1][1])

# Plot bars using calculated bin edges (bars will touch)
for i in range(len(sorted_x)):
    width = bin_edges[i+1] - bin_edges[i] if i+1 < len(bin_edges) else (sorted_ranges[i][1] - sorted_ranges[i][0])
    left = bin_edges[i]
    label = 'Data (freq)' if i == 0 else ''
    ax.bar(left, sorted_freqs[i], width=width, alpha=0.6, edgecolor='black', align='edge', label=label, linewidth=0.5)

ax.plot(x_plot, pdf_scaled, 'r-', linewidth=2.5, label=f'{selected_dist} fit')

if fitting_mode == "Manual":
    shape_auto, loc_auto_plot, scale_auto_plot = split_params(fitted_params)
    scale_auto_plot = max(float(scale_auto_plot), 1e-6)
    try:
        if len(shape_auto) > 0:
            shape_auto = [max(float(s), 1e-6) if s <= 0 else float(s) for s in shape_auto]
            dist_auto = dist_obj(*shape_auto, loc=loc_auto_plot, scale=scale_auto_plot)
        else:
            dist_auto = dist_obj(loc=loc_auto_plot, scale=scale_auto_plot)
    except:
        try:
            params_list = list(fitted_params)
            if len(params_list) >= 2:
                params_list[-1] = max(float(params_list[-1]), 1e-6)
            dist_auto = dist_obj(*params_list)
        except:
            dist_auto = dist_obj(loc=loc_auto_plot, scale=scale_auto_plot)
    
    try:
        pdf_auto = dist_auto.pdf(x_plot)
        pdf_auto = np.nan_to_num(pdf_auto, nan=0.0, posinf=0.0, neginf=0.0)
        pdf_auto = np.maximum(pdf_auto, 0)
        # Use same scaling approach as main PDF (ranges only)
        pdf_auto_scaled = np.zeros_like(pdf_auto)
        for i, (low, high) in enumerate(ranges):
            bin_width = high - low if high > low else x_range / max(len(x_vals), 1)
            mask = (x_plot >= low) & (x_plot < high)
            if i == len(ranges) - 1:
                mask = mask | (x_plot == high)
            if np.any(mask):
                pdf_auto_scaled[mask] = pdf_auto[mask] * bin_width * total_freq
        uncovered = pdf_auto_scaled == 0
        if np.any(uncovered):
            pdf_auto_scaled[uncovered] = pdf_auto[uncovered] * avg_bin_width * total_freq
        pdf_auto = pdf_auto_scaled
        ax.plot(x_plot, pdf_auto, 'g--', linewidth=2, alpha=0.7, label='Automatic fit')
    except:
        pass

ax.set_xlabel('x (Value)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'{selected_dist} Distribution Fit', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Auto-scale axes (ranges only)
x_min_plot = min([low for low, high in ranges])
x_max_plot = max([high for low, high in ranges])
x_range_plot = x_max_plot - x_min_plot if x_max_plot > x_min_plot else 1.0

x_padding = max(0.1 * x_range_plot, (x_max_plot - x_min_plot) * 0.05) if x_range_plot > 0 else 1.0
ax.set_xlim(x_min_plot - x_padding, x_max_plot + x_padding)
y_max = max(np.max(freqs) if len(freqs) > 0 else 0, 
            np.max(pdf_scaled) if len(pdf_scaled) > 0 else 0) * 1.15
ax.set_ylim(0, max(y_max, 0.01))

st.pyplot(fig)

# Parameters and metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parameters")
    shape_active, loc_active, scale_active = split_params(active_params)
    if len(shape_active) > 0:
        for i, val in enumerate(shape_active):
            st.write(f"Shape {i+1} = `{val:.6f}`")
    st.write(f"Location = `{loc_active:.6f}`")
    st.write(f"Scale = `{scale_active:.6f}`")

with col2:
    st.subheader("Fit Quality")
    mae, max_err = compute_metrics(x_vals, freqs, dist_obj, active_params)
    if not np.isnan(mae):
        st.metric("MAE", f"{mae:.6f}")
        st.metric("Max Error", f"{max_err:.6f}")
