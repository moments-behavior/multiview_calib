import streamlit as st
import subprocess
import os
import tempfile
import time

# --- Configuration ---
SCRIPTS = [
    ("1. charuco_intrinsics", "charuco_intrinsics.py"),
    ("2. format_for_calibration", "format_for_calibration.py"),
    ("3. compute_relative_poses", "compute_relative_poses.py"),
    ("4. concatenate_relative_poses", "concatenate_relative_poses.py"),
    ("5. bundle_adjustment", "bundle_adjustment.py"),
    ("6. global_registration", "global_registration.py"),
    ("7. visualize", "visualize.py"),
]

# --- Streamlit App Layout ---

st.set_page_config(page_title="Multiview Calib", layout="wide")
st.title("Multiview Calibration Runner")

# Initialize session state to hold our data
if 'log_output' not in st.session_state:
    st.session_state.log_output = ""
if 'script_selections' not in st.session_state:
    # Default to the first script being selected
    st.session_state.script_selections = {script[1]: (i == 0) for i, script in enumerate(SCRIPTS)}

# --- Helper Function to Run Scripts ---
def run_script(script_path, config_path, log_placeholder):
    """Runs a single script and streams its output to the placeholder."""
    st.session_state.log_output += f"▶️ Running {os.path.basename(script_path)}...\n"
    log_placeholder.code(st.session_state.log_output, language='log')

    try:
        process = subprocess.Popen(
            ["python", "-u", script_path, "-c", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            st.session_state.log_output += line
            log_placeholder.code(st.session_state.log_output, language='log')
        
        process.wait() # Wait for the process to finish

        if process.returncode == 0:
            st.session_state.log_output += f"✅ {os.path.basename(script_path)} completed successfully.\n\n"
            log_placeholder.code(st.session_state.log_output, language='log')
            return True
        else:
            st.session_state.log_output += f"❌ {os.path.basename(script_path)} failed with exit code {process.returncode}.\n\n"
            log_placeholder.code(st.session_state.log_output, language='log')
            return False

    except Exception as e:
        st.session_state.log_output += f"🔥 An error occurred while running {os.path.basename(script_path)}: {e}\n\n"
        log_placeholder.code(st.session_state.log_output, language='log')
        return False

# --- UI Components ---

# Create two columns for controls and logs
col1, col2 = st.columns([1, 2])

with col1:
    # st.header("Controls")

    # 1. Ask for the experiment directory path
    st.subheader("1. Select Experiment Folder")
    default_path = "/home/ro/exp/calib/2025_08_25_v2/"
    config_dir = st.text_input(
        "Enter the absolute path to the folder containing 'config/config.json'",
        default_path
    )

    config_path = os.path.join(config_dir,"config", "config.json")

    if not os.path.exists(config_path):
        st.error(f"Error: 'config.json' not found in the specified directory: {config_dir}")
        st.stop()
    else:
        st.success(f"✅ Found config file: {config_path}")


    # 2. Script Selection
    st.subheader("2. Select Scripts to Run")
    for label, script_file in SCRIPTS:
        st.session_state.script_selections[script_file] = st.checkbox(
            label, value=st.session_state.script_selections[script_file], key=script_file
        )

    # 3. Run Button
    run_button = st.button("🚀 Run Selected Scripts", type="primary", use_container_width=True)

with col2:
    st.header("Live Log Output")
    log_placeholder = st.empty()
    log_placeholder.code(st.session_state.log_output or "Logs will appear here...", language='log')


# --- Main Logic ---
if run_button:
    if not config_dir or not os.path.exists(config_path):
        st.warning("Please provide a valid directory containing a config.json file.")
    else:
        # Clear previous logs
        st.session_state.log_output = ""
        
        st.session_state.log_output += f"Using config file: {config_path}\n\n"
        
        # Run scripts sequentially
        for label, script_file in SCRIPTS:
            if st.session_state.script_selections.get(script_file):
                # Pass the direct config_path, no temp file needed
                success = run_script(script_file, config_path, log_placeholder)
                if not success:
                    st.error(f"Execution stopped due to failure in {label}.")
                    break # Stop on failure
        
        st.session_state.log_output += "\n✨ All selected scripts have been processed."
        log_placeholder.code(st.session_state.log_output, language='log')
        st.toast("Run complete!")
