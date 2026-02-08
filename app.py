import streamlit as st
import os
import numpy as np
import nibabel as nib
from skimage import measure
import plotly.graph_objects as go
from scipy import ndimage
import io

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

# Initialize session state
if 'patients' not in st.session_state:
    st.session_state['patients'] = {}

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
def load_nifti_from_bytes(file_bytes):
    """Load NIfTI from bytes (uploaded file) - pure in-memory"""
    import gzip
    
    # Decompress gzip if needed
    try:
        # Try to decompress (for .nii.gz files)
        decompressed = gzip.decompress(file_bytes)
        file_obj = io.BytesIO(decompressed)
    except:
        # Not gzipped, use as-is (for .nii files)
        file_obj = io.BytesIO(file_bytes)
    
    # Load using nibabel's FileHolder
    fh = nib.FileHolder(fileobj=file_obj)
    img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    return img.get_fdata()

@st.cache_data
def precompute_percentiles(volume):
    """Precompute intensity percentiles for faster rendering"""
    return np.percentile(volume, [1, 99])

def get_patients():
    """Get list of patient names"""
    return list(st.session_state['patients'].keys())

def get_scans(patient):
    """Get list of scans for a patient"""
    if patient not in st.session_state['patients']:
        return []
    return list(st.session_state['patients'][patient].keys())

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
        label_data = (label_downsampled == lid).astype(np.uint8)
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
            
            x, y, z = verts.T
            i, j, k = faces.T
        except:
            # Skip if marching cubes fails
            continue

        # Choose color based on mode
        if grayscale:
            color = "rgb(180, 180, 180)"
        else:
            color_info = get_label_color(lid)
            rgb = color_info["rgb"]
            color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        label_name = label_names.get(lid, f"Label {lid}")

        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=0.85,
            name=label_name,
            flatshading=False,
            lighting=dict(
                ambient=0.7,
                diffuse=1.0,
                specular=1.2,
                roughness=0.3,
                fresnel=0.5
            ),
            lightposition=dict(x=1000, y=1000, z=1000),
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
    else:
        scene_bg = "rgb(17, 17, 17)"
        paper_bg = "rgb(17, 17, 17)"
        axis_bg = "rgb(40, 40, 40)"
    
    fig = go.Figure(data=meshes)
    
    layout_config = dict(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False, showbackground=True, backgroundcolor=axis_bg),
            yaxis=dict(visible=False, showbackground=True, backgroundcolor=axis_bg),
            zaxis=dict(visible=False, showbackground=True, backgroundcolor=axis_bg),
            bgcolor=scene_bg,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            dragmode='orbit'
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
    
    fig.update_layout(layout_config)
    return fig


# ---------------- MAIN APP ----------------
def main_app():
    st.title("🫀 Medical Scan Viewer")

    st.sidebar.header("Patients")

    patient_name = st.sidebar.text_input("New Patient Name")

    if st.sidebar.button("Create Patient"):
        if patient_name:
            if patient_name not in st.session_state['patients']:
                st.session_state['patients'][patient_name] = {}
                st.sidebar.success(f"Created patient: {patient_name}")
                st.rerun()
            else:
                st.sidebar.warning(f"Patient '{patient_name}' already exists")

    patients = get_patients()
    
    # Check if there are any patients
    if not patients:
        st.info("👋 Welcome! To get started:")
        st.markdown("""
        1. **Create a patient** using the sidebar (enter a name and click 'Create Patient')
        2. **Select the patient** from the dropdown
        3. **Upload scan files** (.nii.gz format)
        4. **View and analyze** the medical scans!
        
        ⚠️ **Note:** Files are stored in browser session only and will be lost when you refresh or close the page.
        """)
        st.stop()
    
    selected_patient = st.sidebar.selectbox("Select Patient", patients)
    
    if not selected_patient:
        st.warning("⚠️ Please select a patient from the dropdown")
        st.stop()

    st.sidebar.header("Upload Scan")
    scan_file = st.sidebar.file_uploader("Upload CT/MRI (_image.nii.gz)", type=['gz'])
    label_file = st.sidebar.file_uploader("Upload Label (_label.nii.gz)", type=['gz'])

    if st.sidebar.button("Save Scan"):
        if scan_file and label_file:
            try:
                # Read file bytes
                scan_bytes = scan_file.read()
                label_bytes = label_file.read()
                
                # Create scan name
                scan_name = scan_file.name.replace('_image.nii.gz', '').replace('.nii.gz', '')
                
                # Store in session state
                if selected_patient not in st.session_state['patients']:
                    st.session_state['patients'][selected_patient] = {}
                
                st.session_state['patients'][selected_patient][scan_name] = {
                    'image_bytes': scan_bytes,
                    'label_bytes': label_bytes,
                    'image_name': scan_file.name,
                    'label_name': label_file.name
                }
                
                st.sidebar.success(f"✅ Saved scan: {scan_name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error saving scan: {str(e)}")
        else:
            st.sidebar.error("Please upload both image and label files")

    scans = get_scans(selected_patient)
    
    if not scans:
        st.warning(f"No scans found for patient '{selected_patient}'. Please upload scan files using the sidebar.")
        st.stop()
    
    scan_choice = st.selectbox("Select Scan", scans)

    if scan_choice:
        try:
            # Load scan data from session state
            scan_data = st.session_state['patients'][selected_patient][scan_choice]
            
            # Load volumes from bytes
            volume = load_nifti_from_bytes(scan_data['image_bytes'])
            label = load_nifti_from_bytes(scan_data['label_bytes']).astype(np.int32)

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
        
        except Exception as e:
            st.error(f"Error loading scan: {str(e)}")
            st.info("Try re-uploading the scan files.")


# ---------------- ROUTER ----------------
if "user" not in st.session_state:
    login_page()
else:
    main_app()
