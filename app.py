import streamlit as st
import os
import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go
from scipy import ndimage

DATA_DIR = "patients"
os.makedirs(DATA_DIR, exist_ok=True)

# Consistent color map for labels
COLOR_MAP = {
    1: {"rgb": [255, 0, 0], "name": "red"},
    2: {"rgb": [0, 255, 0], "name": "green"},
    3: {"rgb": [0, 0, 255], "name": "blue"},
    4: {"rgb": [255, 255, 0], "name": "yellow"},
    5: {"rgb": [255, 0, 255], "name": "magenta"},
    6: {"rgb": [0, 255, 255], "name": "cyan"},
    7: {"rgb": [255, 165, 0], "name": "orange"},
    205: {"rgb": [255, 0, 255], "name": "magenta"},
    420: {"rgb": [0, 0, 255], "name": "blue"},
    500: {"rgb": [255, 0, 0], "name": "red"},
    550: {"rgb": [255, 255, 0], "name": "yellow"},
    600: {"rgb": [0, 255, 0], "name": "green"},
    820: {"rgb": [0, 255, 255], "name": "cyan"},
    850: {"rgb": [255, 165, 0], "name": "orange"},
}

def get_label_color(label_id):
    """Get consistent color for a label ID"""
    if label_id in COLOR_MAP:
        return COLOR_MAP[label_id]
    # Generate consistent random color for labels beyond predefined ones
    np.random.seed(int(label_id))
    rgb = np.random.randint(50, 255, size=3).tolist()
    return {"rgb": rgb, "name": f"color_{label_id}"}

# ---------------- AUTH ----------------
def login_page():
    st.title("Doctor Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state["user"] = username
            st.rerun()
        else:
            st.error("Enter username and password")


# ---------------- HELPERS ----------------
@st.cache_data
def load_nifti(path):
    return nib.load(path).get_fdata()

@st.cache_data
def precompute_percentiles(volume):
    """Precompute intensity percentiles for faster rendering"""
    return np.percentile(volume, [1, 99])

@st.cache_data
def render_slice_cached(volume_shape, label_shape, view, idx, vmin, vmax, opacity, volume_hash, label_hash):
    """Cached version of render_slice - uses hashes to determine if data changed"""
    # This gets called by render_slice with actual data
    pass

def get_patients():
    return os.listdir(DATA_DIR)

def get_scans(patient):
    folder = os.path.join(DATA_DIR, patient)
    return [f for f in os.listdir(folder) if "_image" in f]

def render_slice(volume, label, view, idx, vmin, vmax, opacity):
    """Optimized slice rendering with minimal operations"""
    # Extract slice based on view
    if view == "axial":
        img = volume[idx, :, :]
        lab = label[idx, :, :]
    elif view == "coronal":
        img = volume[:, idx, :]
        lab = label[:, idx, :]
    else:
        img = volume[:, :, idx]
        lab = label[:, :, idx]

    # Faster normalization using vectorized operations
    norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1).copy()

    # Get unique labels once
    unique_labels = np.unique(lab[lab != 0])
    
    # Vectorized color overlay
    for lid in unique_labels:
        color_info = get_label_color(lid)
        color = np.array(color_info["rgb"], dtype=np.float32)
        mask = lab == lid
        rgb[mask] = ((1 - opacity) * rgb[mask] + opacity * color).astype(np.uint8)

    return rgb


def render_3d(label_volume, selected_labels, label_names, grayscale=False, bg_style="dark"):
    from scipy import ndimage
    import hashlib
    import pickle
    import base64
    import os
    
    # Create cache directory
    cache_dir = os.path.join(DATA_DIR, ".mesh_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load predetermined background image if exists
    custom_bg_image = None
    bg_image_path = "background.jpg"  # Place your background image here
    if os.path.exists(bg_image_path):
        try:
            with open(bg_image_path, "rb") as img_file:
                custom_bg_image = base64.b64encode(img_file.read()).decode()
        except:
            custom_bg_image = None
    
    meshes = []

    # Downsample for faster processing if volume is large
    original_shape = label_volume.shape
    max_dim = max(original_shape)
    
    if max_dim > 256:
        # Calculate downsampling factor
        factor = max_dim / 256
        zoom_factors = [1/factor, 1/factor, 1/factor]
        # Use nearest neighbor to preserve label values
        label_downsampled = ndimage.zoom(label_volume, zoom_factors, order=0)
    else:
        label_downsampled = label_volume
        zoom_factors = [1, 1, 1]

    for lid in selected_labels:
        # Create cache key based on label data
        label_data = (label_downsampled == lid).astype(np.uint8)
        cache_key = hashlib.md5(label_data.tobytes()).hexdigest()
        cache_file = os.path.join(cache_dir, f"mesh_{lid}_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_mesh_data = pickle.load(f)
                x, y, z = cached_mesh_data['verts'].T
                i, j, k = cached_mesh_data['faces'].T
            except:
                # Cache corrupted, regenerate
                cached_mesh_data = None
        else:
            cached_mesh_data = None
        
        # Generate mesh if not cached
        if cached_mesh_data is None:
            binary = label_data
            
            # Skip if very small region
            if binary.sum() < 50:
                continue

            try:
                # Use step_size for faster marching cubes
                verts, faces, _, _ = measure.marching_cubes(binary, level=0.5, step_size=2)
                
                # Scale vertices back to original size if downsampled
                if max_dim > 256:
                    verts = verts * np.array([zoom_factors[0]**-1, zoom_factors[1]**-1, zoom_factors[2]**-1])
                
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump({'verts': verts, 'faces': faces}, f)
                
                x, y, z = verts.T
                i, j, k = faces.T
            except:
                # Skip if marching cubes fails
                continue

        # Choose color based on mode
        if grayscale:
            # Use grayscale - all structures in gray
            color = "rgb(180, 180, 180)"
        else:
            # Use label-specific colors
            color_info = get_label_color(lid)
            rgb = color_info["rgb"]
            color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        label_name = label_names.get(lid, f"Label {lid}")

        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=0.85,  # More opaque for better visibility
            name=label_name,
            flatshading=False,  # Smooth shading for better VR effect
            lighting=dict(
                ambient=0.7,      # Increased ambient light
                diffuse=1.0,      # Full diffuse lighting
                specular=1.2,     # Enhanced specular highlights
                roughness=0.3,    # Smoother surface
                fresnel=0.5       # Edge highlighting for depth
            ),
            lightposition=dict(
                x=1000,
                y=1000, 
                z=1000
            ),
            hoverinfo='name',
            hovertemplate='<b>%{fullData.name}</b><extra></extra>'
        )

        meshes.append(mesh)

    # Define background colors based on style
    if bg_style == "dark":
        scene_bg = "rgb(17, 17, 17)"
        paper_bg = "rgb(17, 17, 17)"
        axis_bg = "rgb(40, 40, 40)"
    elif bg_style == "light":
        scene_bg = "rgb(240, 240, 240)"
        paper_bg = "rgb(255, 255, 255)"
        axis_bg = "rgb(220, 220, 220)"
    elif bg_style == "custom" and custom_bg_image:
        # Use semi-transparent backgrounds so custom image shows through
        scene_bg = "rgba(0, 0, 0, 0.3)"
        paper_bg = "rgba(0, 0, 0, 0.3)"
        axis_bg = "rgba(255, 255, 255, 0.05)"
    else:
        scene_bg = "rgb(17, 17, 17)"
        paper_bg = "rgb(17, 17, 17)"
        axis_bg = "rgb(40, 40, 40)"
    
    # No grid lines - just the heart meshes
    fig = go.Figure(data=meshes)
    
    # Build layout configuration optimized for VR-like rotation
    layout_config = dict(
        scene=dict(
            aspectmode="data",
            xaxis=dict(
                visible=False,
                showbackground=True,
                backgroundcolor=axis_bg
            ),
            yaxis=dict(
                visible=False,
                showbackground=True,
                backgroundcolor=axis_bg
            ),
            zaxis=dict(
                visible=False,
                showbackground=True,
                backgroundcolor=axis_bg
            ),
            bgcolor=scene_bg,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            # Enhanced rotation settings for smooth VR-like experience
            dragmode='orbit'  # Allows free rotation around the heart
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        paper_bgcolor=paper_bg,
    )
    
    # Add custom background image if provided
    if bg_style == "custom" and custom_bg_image:
        layout_config['images'] = [dict(
            source=f"data:image/png;base64,{custom_bg_image}",
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="stretch",
            opacity=0.4,
            layer="below"
        )]
    
    fig.update_layout(layout_config)

    return fig


# ---------------- MAIN APP ----------------
def main_app():
    st.title("🫀 Medical Scan Viewer")

    st.sidebar.header("Patients")

    patient = st.sidebar.text_input("New Patient Name")

    if st.sidebar.button("Create Patient"):
        if patient:
            os.makedirs(os.path.join(DATA_DIR, patient), exist_ok=True)

    patients = get_patients()
    selected_patient = st.sidebar.selectbox("Select Patient", patients)

    st.sidebar.header("Upload Scan")
    scan_file = st.sidebar.file_uploader("Upload CT/MRI (_image.nii.gz)")
    label_file = st.sidebar.file_uploader("Upload Label (_label.nii.gz)")

    if st.sidebar.button("Save Scan"):
        if scan_file and label_file:
            pdir = os.path.join(DATA_DIR, selected_patient)
            with open(os.path.join(pdir, scan_file.name), "wb") as f:
                f.write(scan_file.getbuffer())
            with open(os.path.join(pdir, label_file.name), "wb") as f:
                f.write(label_file.getbuffer())
            st.sidebar.success("Saved!")

    scans = get_scans(selected_patient)
    scan_choice = st.selectbox("Select Scan", scans)

    if scan_choice:
        image_path = os.path.join(DATA_DIR, selected_patient, scan_choice)
        label_path = image_path.replace("_image", "_label")

        volume = load_nifti(image_path)
        label = load_nifti(label_path).astype(np.int32)

        st.write("Volume shape:", volume.shape)

        z, y, x = volume.shape
        max_slice = min(z, y, x) - 1

        slice_idx = st.slider("Slice Index", 0, max_slice, max_slice // 2)
        opacity = st.slider("Label Opacity", 0.0, 1.0, 0.5)

        # Precompute percentiles once for faster rendering
        vmin, vmax = precompute_percentiles(volume)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.text("Axial")
            img = render_slice(volume, label, "axial", slice_idx, vmin, vmax, opacity)
            st.image(img, use_container_width=True)

        with col2:
            st.text("Coronal")
            img = render_slice(volume, label, "coronal", slice_idx, vmin, vmax, opacity)
            st.image(img, use_container_width=True)

        with col3:
            st.text("Sagittal")
            img = render_slice(volume, label, "sagittal", slice_idx, vmin, vmax, opacity)
            st.image(img, use_container_width=True)

        # Label Legend
        st.markdown("---")
        st.subheader("Label Legend")
        
        # Get unique labels in the current data
        unique_labels_in_data = np.unique(label)
        unique_labels_in_data = unique_labels_in_data[unique_labels_in_data != 0]
        
        # Label definitions
        LABEL_NAMES = {
            500: "LV - Left Ventricle Blood Cavity",
            600: "RV - Right Ventricle Blood Cavity",
            420: "LA - Left Atrium Blood Cavity",
            550: "RA - Right Atrium Blood Cavity",
            205: "MYO - Myocardium of the Left Ventricle",
            820: "AA - Ascending Aorta",
            850: "PA - Pulmonary Artery"
        }
        
        # Display legend in columns
        legend_cols = st.columns(3)
        for idx, lid in enumerate(unique_labels_in_data):
            col = legend_cols[idx % 3]
            with col:
                color_info = get_label_color(lid)
                rgb = color_info["rgb"]
                hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                label_name = LABEL_NAMES.get(lid, f"Label {lid}")
                
                # Display colored box with label name
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin-bottom: 8px;">'
                    f'<div style="width: 20px; height: 20px; background-color: {hex_color}; '
                    f'margin-right: 10px; border: 1px solid #ccc;"></div>'
                    f'<span><b>{lid}</b>: {label_name}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        st.markdown("---")

        st.subheader("3D View")
        
        # Background style selector
        st.write("**Background Style:**")
        bg_col1, bg_col2, bg_col3 = st.columns(3)
        with bg_col1:
            if st.button("🌑 Dark (VR)"):
                st.session_state["bg_style"] = "dark"
        with bg_col2:
            if st.button("⚪ Light"):
                st.session_state["bg_style"] = "light"
        with bg_col3:
            if st.button("🖼️ Medical Room"):
                st.session_state["bg_style"] = "custom"
        
        st.markdown("---")

        # Button options for 3D rendering
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Show in 3D (Colored)"):
                st.session_state["show_3d"] = True
                st.session_state["grayscale_3d"] = False
        
        with col_btn2:
            if st.button("Show in 3D (Grayscale)"):
                st.session_state["show_3d"] = True
                st.session_state["grayscale_3d"] = True

        if st.session_state.get("show_3d", False):
            # Get unique labels present in the data
            unique_labels = np.unique(label)
            available_labels = [lid for lid in unique_labels if lid != 0]

            # Label definitions
            LABEL_NAMES = {
                500: "Left Ventricle Blood Cavity (LV)",
                600: "Right Ventricle Blood Cavity (RV)",
                420: "Left Atrium Blood Cavity (LA)",
                550: "Right Atrium Blood Cavity (RA)",
                205: "Myocardium of the Left Ventricle (MYO)",
                820: "Ascending Aorta (AA)",
                850: "Pulmonary Artery (PA)"
            }

            # Create toggle options for each available label
            st.write("**Select labels to display:**")
            selected_labels = []
            
            cols = st.columns(4)
            for idx, lid in enumerate(available_labels):
                col = cols[idx % 4]
                with col:
                    label_name = LABEL_NAMES.get(lid, f"Label {lid}")
                    if st.checkbox(label_name, value=True, key=f"label_{lid}"):
                        selected_labels.append(lid)

            if selected_labels:
                grayscale_mode = st.session_state.get("grayscale_3d", False)
                bg_style = st.session_state.get("bg_style", "dark")
                with st.spinner("Generating 3D mesh..."):
                    fig = render_3d(label, selected_labels, LABEL_NAMES, grayscale=grayscale_mode, bg_style=bg_style)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one label to display.")


# ---------------- ROUTER ----------------
if "user" not in st.session_state:
    login_page()
else:
    main_app()