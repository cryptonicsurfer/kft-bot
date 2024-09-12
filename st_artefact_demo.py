import streamlit as st

# Initialize session state for artifacts
if 'artifacts' not in st.session_state:
    st.session_state.artifacts = {}

def create_artifact(identifier, content, title):
    st.session_state.artifacts[identifier] = {
        'content': content,
        'title': title
    }

def display_artifact(identifier):
    artifact = st.session_state.artifacts.get(identifier)
    if artifact:
        with st.sidebar:
            st.title(artifact['title'])
            st.code(artifact['content'])
    else:
        st.sidebar.write("No artifact selected")

# Main page content
st.title("Streamlit Artifact-like System")

# Create some example artifacts
create_artifact('example-code', 'print("Hello, World!")', 'Example Code')
create_artifact('custom-function', 'def greet(name):\n    return f"Hello, {name}!"', 'Custom Function')

# Create buttons for each artifact
for artifact_id, artifact_data in st.session_state.artifacts.items():
    if st.button(f"Display {artifact_data['title']}"):
        display_artifact(artifact_id)

# This will display the last selected artifact by default
if 'last_selected_artifact' in st.session_state:
    display_artifact(st.session_state.last_selected_artifact)

# Main content
st.write("This is the main content of the page.")
st.write("Select an artifact from the dropdown and click the button to display it in the sidebar.")

# Option to create new artifact
new_artifact_id = st.text_input("New Artifact Identifier:")
new_artifact_title = st.text_input("New Artifact Title:")
new_artifact_content = st.text_area("New Artifact Content:")

if st.button("Create New Artifact"):
    if new_artifact_id and new_artifact_title and new_artifact_content:
        create_artifact(new_artifact_id, new_artifact_content, new_artifact_title)
        st.success(f"New artifact '{new_artifact_id}' created!")
    else:
        st.error("Please fill in all fields to create a new artifact.")
