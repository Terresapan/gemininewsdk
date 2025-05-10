import streamlit as st
import os
from main import GeminiChatBackend
import tempfile

# Set up page configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="wide")

# Initialize session states for storing conversation histories
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []

if "gemini_backend" not in st.session_state:
    st.session_state.gemini_backend = None
    st.session_state.chat_initialized = False
    st.session_state.search_enabled = False
    st.session_state.code_execution_enabled = False
    st.session_state.api_key = ""
    st.session_state.model_id = "gemini-2.5-flash"

def init_backend():
    """Initialize the backend if it's not already initialized"""
    try:
        # Use the API key from session state (user input)
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return False
            
        backend = GeminiChatBackend(api_key=api_key)
        
        # Define the system instruction
        system_instruction = """
        You are a helpful assistant and always respond in a friendly and informative manner.
        When search is enabled, you can look up current information to provide more accurate responses.
        When code execution is enabled, you can write and execute code to solve problems.
        You can also analyze PDF, CSV, text files, images, audio, and video files when they are uploaded.
        """
        
        # Get the current settings
        search_enabled = st.session_state.get("search_enabled", False)
        code_execution_enabled = st.session_state.get("code_execution_enabled", False)
        model_id = st.session_state.get("model_id", "gemini-2.5-flash")
        
        # Initialize the chat
        if backend.initialize_chat(
            model_id=model_id,
            system_instruction=system_instruction,
            temperature=0.5,
            search_enabled=search_enabled,
            code_execution_enabled=code_execution_enabled
        ):
            st.session_state.gemini_backend = backend
            st.session_state.chat_initialized = True
            return True
        else:
            st.error("Failed to initialize chat")
            return False
    except Exception as e:
        st.error(f"Error initializing backend: {e}")
        return False

# App title
st.title("Gemini2.5 New SDK Test")

# Create tabs for different functionalities
chat_tab, file_tab, data_analytics_tab, image_tab = st.tabs(["Chat Mode", "File Analysis Mode", "Data Analytics Mode", "Image Generation Mode"])

# Sidebar for global settings
with st.sidebar:
    st.title("Settings")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("api_key", ""),
        type="password",
        help="Enter your Gemini API key here"
    )
    
    # Update API key in session state if changed
    if api_key != st.session_state.get("api_key", ""):
        st.session_state.api_key = api_key
        st.session_state.chat_initialized = False
    
    # Model selection
    model_options = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro-preview-05-06"]
    selected_model = st.selectbox(
        "Select Gemini Model",
        options=model_options,
        index=model_options.index(st.session_state.get("model_id", "gemini-2.0-flash", "gemini-2.5-flash")),
        help="Choose which Gemini model to use"
    )
    
    # Update model in session state if changed
    if selected_model != st.session_state.get("model_id", "gemini-2.5-flash"):
        st.session_state.model_id = selected_model
        st.session_state.chat_initialized = False
    
    # Search toggle for chat mode
    search_enabled = st.toggle(
        "Enable Web Search",
        value=st.session_state.get("search_enabled", False),
        help="Enable web search grounding to provide up-to-date information"
    )
    
    # Code execution toggle for chat mode
    code_execution_enabled = st.toggle(
        "Enable Code Execution",
        value=st.session_state.get("code_execution_enabled", False),
        help="Enable code execution to run Python code in responses"
    )
    
    # Make search and code execution mutually exclusive
    if search_enabled and code_execution_enabled:
        st.warning("Search and Code Execution cannot be enabled simultaneously. Disabling the other option.")
    
    # Update search setting if changed
    if search_enabled != st.session_state.get("search_enabled", False):
        st.session_state.search_enabled = search_enabled
        
        # If enabling search, disable code execution
        if search_enabled and st.session_state.get("code_execution_enabled", False):
            st.session_state.code_execution_enabled = False
            code_execution_enabled = False
        
        # Update backend if it exists
        if st.session_state.gemini_backend is not None:
            st.session_state.gemini_backend.toggle_search(search_enabled)
            
            # Show appropriate message
            if search_enabled:
                st.sidebar.success("Web search enabled! The chatbot can now search for current information.")
            else:
                st.sidebar.info("Web search disabled. The chatbot will rely on its training data only.")
    
    # Update code execution setting if changed
    if code_execution_enabled != st.session_state.get("code_execution_enabled", False):
        st.session_state.code_execution_enabled = code_execution_enabled
        
        # If enabling code execution, disable search
        if code_execution_enabled and st.session_state.get("search_enabled", False):
            st.session_state.search_enabled = False
            search_enabled = False
        
        # Update backend if it exists
        if st.session_state.gemini_backend is not None:
            st.session_state.gemini_backend.toggle_code_execution(code_execution_enabled)
            
            # Show appropriate message
            if code_execution_enabled:
                st.sidebar.success("Code execution enabled! The chatbot can now write and run code.")
            else:
                st.sidebar.info("Code execution disabled.")
    
    # Reinitialize button
    if st.button("Reinitialize Chat"):
        st.session_state.chat_initialized = False
        st.rerun()

# Initialize backend automatically if not already done
if not st.session_state.chat_initialized and st.session_state.get("api_key", ""):
    with st.spinner("Initializing chat..."):
        if init_backend():
            st.success("Chat initialized successfully!")

# CHAT TAB
with chat_tab:
    st.header("Chat Mode")
    st.write("Ask questions and get answers from Gemini")
    
    # Status indicators
    if st.session_state.get("search_enabled", False):
        st.info("🔍 Web search is enabled. Ask me about current events or topics that may require up-to-date information.")
    elif st.session_state.get("code_execution_enabled", False):
        st.info("💻 Code execution is enabled. Ask me to write and run code to solve problems.")
    
    # Create a container for chat messages
    chat_container = st.container(height=600)

    # Add a button to clear chat history at the bottom
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Accept user input - this will now appear at the bottom
    if st.session_state.chat_initialized:
        user_input = st.chat_input("Ask something...", key="chat_input")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            try:
                # Send message to Gemini
                spinner_text = "Thinking..."
                if st.session_state.get("search_enabled", False):
                    spinner_text += " (searching the web)"
                elif st.session_state.get("code_execution_enabled", False):
                    spinner_text += " (executing code)"
                    
                with st.spinner(spinner_text):
                    backend = st.session_state.gemini_backend
                    response = backend.send_message(user_input)
                
                # Process the response based on whether code execution is enabled
                if st.session_state.get("code_execution_enabled", False) and isinstance(response, dict):
                    # Format the response for code execution
                    formatted_response = ""
                    
                    # Add text parts
                    for text in response["text"]:
                        formatted_response += text + "\n\n"
                    
                    # Add executable code parts
                    for code in response["executable_code"]:
                        formatted_response += f"```python\n{code}\n```\n\n"
                    
                    # Add code execution results
                    for result in response["code_execution_result"]:
                        formatted_response += f"**Execution Result:**\n```\n{result}\n```\n\n"
                    
                    # Add inline data if any (not handling image display here)
                    if response["inline_data"]:
                        formatted_response += "*Note: Some inline data (like images) may not be displayed properly.*\n\n"
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": formatted_response.strip()})
                else:
                    # Add regular text response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                
                # Force a rerun to update the chat display with new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting response: {e}")
    else:
        st.warning("Chat initialization failed. Please check your API key in the Streamlit secrets.")
    
    # Display chat messages from history inside the container
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# FILE ANALYSIS TAB
with file_tab:
    st.header("File Analysis Mode")
    st.write("Upload PDF, CSV, TXT, audio, or video files to ask questions about them")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload and controls
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "csv", "txt", "mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "gif", "webp"], accept_multiple_files=True, key="file_uploader")
        
        if uploaded_files:
            st.session_state.pdf_files = uploaded_files  # Keep the same session state key for backward compatibility
            
            # Display uploaded files with file type indicators
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            for i, file in enumerate(uploaded_files):
                if file.name.lower().endswith(".pdf"):
                    file_type = "📄 PDF"
                elif file.name.lower().endswith(".csv"):
                    file_type = "📊 CSV"
                elif file.name.lower().endswith(".txt"):
                    file_type = "📝 TXT"
                elif file.name.lower().endswith((".mp3", ".wav", ".m4a")):
                    file_type = "🔊 Audio"
                elif file.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    file_type = "🎬 Video"
                elif file.name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                    file_type = "🖼️ Image"
                else:
                    file_type = "📁 File"
                st.text(f"{i+1}. {file_type}: {file.name}")
            
            # Query input
            file_query = st.text_area(
                "Ask a question about the files", 
                placeholder="E.g., 'Summarize these documents' or 'Analyze this data'",
                height=100
            )
            
            # Process files button
            if st.button("Process Files", key="process_files") and file_query:
                if not st.session_state.chat_initialized:
                    st.error("Chat not initialized. Please initialize the chat first.")
                else:
                    with st.spinner("Processing files..."):
                        try:
                            # Save uploaded files to temporary files with appropriate extensions
                            temp_files = []
                            file_types = []
                            
                            for uploaded_file in uploaded_files:
                                # Determine file extension and type
                                if uploaded_file.name.lower().endswith(".pdf"):
                                    file_extension = ".pdf"
                                    file_type = "pdf"
                                elif uploaded_file.name.lower().endswith(".csv"):
                                    file_extension = ".csv"
                                    file_type = "csv"
                                elif uploaded_file.name.lower().endswith(".txt"):
                                    file_extension = ".txt"
                                    file_type = "txt"
                                elif uploaded_file.name.lower().endswith((".mp3", ".wav", ".m4a")):
                                    # Preserve the original extension for audio files
                                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                                    file_type = "audio"
                                elif uploaded_file.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                                    # Preserve the original extension for video files
                                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                                    file_type = "video"
                                elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                    # Preserve the original extension for image files
                                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                                    file_type = "image"
                                else:
                                    # Default case
                                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                                    file_type = "pdf"  # Default to PDF processing
                                
                                # Create temporary file with appropriate extension
                                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_files.append(tmp_file.name)
                                    file_types.append(file_type)
                            
                            # Process the files with the query
                            backend = st.session_state.gemini_backend
                            
                            # Show special message for video files as they may take longer
                            if "video" in file_types:
                                st.info("Processing video files may take a bit longer. Please be patient...")
                                
                            response_text = backend.process_files(temp_files, file_query, file_types=file_types)
                            
                            # Add the interaction to chat history
                            file_names = ", ".join([file.name for file in uploaded_files])
                            file_message = f"📁 Files: {file_names}\n\nQuery: {file_query}"
                            st.session_state.pdf_messages.append({"role": "user", "content": file_message})
                            st.session_state.pdf_messages.append({"role": "assistant", "content": response_text})
                            
                            # Clean up the temporary files
                            for temp_file in temp_files:
                                os.unlink(temp_file)
                            
                            # Rerun to update the chat display
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing files: {e}")
        
        # Add a button to clear file chat history
        if st.button("Clear File Chat History", key="clear_file_chat"):
            st.session_state.pdf_messages = []
            st.rerun()
    
    with col2:
        # Display file chat messages from history
        st.subheader("File Analysis History")
        
        # Create a container with scrolling for the file chat history
        file_chat_container = st.container(height=700, border=True)
        
        with file_chat_container:
            for message in st.session_state.pdf_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# DATA ANALYTICS TAB
with data_analytics_tab:
    st.header("Data Analytics Mode")
    st.write("Upload CSV or Excel files for data analysis with code execution enabled")
    
    # Initialize session state for data analytics messages if not already done
    if "data_analytics_messages" not in st.session_state:
        st.session_state.data_analytics_messages = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File upload and controls
        uploaded_data_files = st.file_uploader("Upload data files", type=["csv", "xlsx"], accept_multiple_files=True, key="data_file_uploader")
        
        if uploaded_data_files:
            # Display uploaded files with file type indicators
            st.success(f"{len(uploaded_data_files)} file(s) uploaded")
            for i, file in enumerate(uploaded_data_files):
                if file.name.lower().endswith(".csv"):
                    file_type = "📊 CSV"
                elif file.name.lower().endswith(".xlsx"):
                    file_type = "📈 Excel"
                st.text(f"{i+1}. {file_type}: {file.name}")
            
            # Query input
            data_query = st.text_area(
                "Ask a question about the data", 
                placeholder="E.g., 'Analyze this dataset' or 'Create a visualization of this data'",
                height=100
            )
            
            # Process files button
            if st.button("Analyze Data", key="analyze_data") and data_query:
                if not st.session_state.chat_initialized:
                    st.error("Chat not initialized. Please initialize the chat first.")
                else:
                    with st.spinner("Analyzing data..."):
                        try:
                            # Save uploaded files to temporary files with appropriate extensions
                            temp_files = []
                            file_types = []
                            
                            for uploaded_file in uploaded_data_files:
                                # Determine file extension and type
                                if uploaded_file.name.lower().endswith(".csv"):
                                    file_extension = ".csv"
                                    file_type = "csv"
                                elif uploaded_file.name.lower().endswith(".xlsx"):
                                    file_extension = ".xlsx"
                                    file_type = "csv"  # Process Excel as CSV for now
                                
                                # Create temporary file with appropriate extension
                                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_files.append(tmp_file.name)
                                    file_types.append(file_type)
                            
                            # Ensure code execution is enabled for data analysis
                            backend = st.session_state.gemini_backend
                            original_code_execution_state = backend.is_code_execution_enabled()
                            backend.toggle_code_execution(True)
                            
                            # Process the files with the query
                            response = backend.process_files(temp_files, data_query, file_types=file_types)
                            
                            # Add the interaction to chat history
                            file_names = ", ".join([file.name for file in uploaded_data_files])
                            file_message = f"📊 Data Files: {file_names}\n\nQuery: {data_query}"
                            st.session_state.data_analytics_messages.append({"role": "user", "content": file_message})
                            st.session_state.data_analytics_messages.append({"role": "assistant", "content": response})
                            
                            # Clean up the temporary files
                            for temp_file in temp_files:
                                os.unlink(temp_file)
                            
                            # Restore original code execution state if needed
                            if not original_code_execution_state:
                                backend.toggle_code_execution(original_code_execution_state)
                            
                            # Rerun to update the chat display
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error analyzing data: {e}")
        
        # Add a button to clear data analytics chat history
        if st.button("Clear Data Analytics History", key="clear_data_analytics"):
            st.session_state.data_analytics_messages = []
            st.rerun()
    
    with col2:
        # Display data analytics chat messages from history
        st.subheader("Data Analysis History")
        
        # Create a container with scrolling for the data analytics chat history
        data_chat_container = st.container(height=700, border=True)
        
        with data_chat_container:
            for message in st.session_state.data_analytics_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

# IMAGE GENERATION TAB
with image_tab:
    st.header("Image Generation Mode")
    st.write("Generate images using Gemini's Imagen model")
    
    # Initialize session state for storing generated images
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image generation controls
        prompt = st.text_area(
            "Enter a prompt for image generation", 
            placeholder="E.g., 'A cat lounging lazily on a sunny windowsill playing with a kid toy.'",
            height=100
        )
        
        # Image generation settings
        number_of_images = st.slider("Number of images", min_value=1, max_value=4, value=1)
        
        person_generation = st.radio(
            "Person generation",
            options=["DONT_ALLOW", "ALLOW_ADULT"],
            index=1,
            horizontal=True
        )
        
        aspect_ratio = st.select_slider(
            "Aspect ratio",
            options=["1:1", "3:4", "4:3", "16:9", "9:16"],
            value="1:1"
        )
        
        # Generate button
        if st.button("Generate Images", key="generate_images") and prompt:
            if not st.session_state.chat_initialized:
                st.error("Chat not initialized. Please initialize the chat first.")
            else:
                with st.spinner("Generating images...this may take a moment"):
                    try:
                        # Generate images using the backend
                        backend = st.session_state.gemini_backend
                        generated_images = backend.generate_images(
                            prompt=prompt,
                            number_of_images=number_of_images,
                            person_generation=person_generation,
                            aspect_ratio=aspect_ratio
                        )
                        
                        # Store the generated images in session state
                        st.session_state.generated_images = generated_images
                        
                        # Rerun to update the display
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating images: {e}")
        
        # Add a button to clear generated images
        if st.button("Clear Generated Images", key="clear_images"):
            st.session_state.generated_images = []
            st.rerun()
    
    with col2:
        # Display generated images
        st.subheader("Generated Images")
        
        # Create a container for displaying images
        image_container = st.container(height=700, border=True)
        
        with image_container:
            if st.session_state.generated_images:
                for i, image_bytes in enumerate(st.session_state.generated_images):
                    # The image_bytes are already extracted in the backend
                    # Display the image using the bytes
                    st.image(image_bytes, caption=f"Image {i+1}", use_container_width=True)
                    
                    # For download button, use the same bytes
                    st.download_button(
                        label=f"Download Image {i+1}",
                        data=image_bytes,
                        file_name=f"generated_image_{i+1}.jpg",
                        mime="image/jpeg"
                    )
                    st.divider()
            else:
                st.info("No images generated yet. Enter a prompt and click 'Generate Images' to create some!")
