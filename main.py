from google import genai
from google.genai import types
import os
import pathlib
import pandas as pd
import io
import time
import tempfile


class GeminiChatBackend:
    def __init__(self, api_key=None):
        """
        Initialize the Gemini chat backend.
        
        Args:
            api_key (str, optional): API key for Gemini. If None, will try to use from environment.
        """
        if api_key:
            self.api_key = api_key
        elif "GOOGLE_API_KEY" in os.environ:
            self.api_key = os.environ["GOOGLE_API_KEY"]
        else:
            raise ValueError("No API key provided and none found in environment")
            
        self.client = genai.Client(api_key=self.api_key)
        self.chat = None
        self.model_id = "gemini-2.5-flash"
        self.search_enabled = False
        self.code_execution_enabled = False
        
    def initialize_chat(self, model_id="gemini-2.5-flash", system_instruction=None, temperature=0.5, search_enabled=False, code_execution_enabled=False):
        """
        Initialize a new chat session.
        
        Args:
            model_id (str): The ID of the Gemini model to use
            system_instruction (str, optional): System instructions for the chat
            temperature (float): Temperature for generating responses (0.0 to 1.0)
            search_enabled (bool): Whether to enable search grounding
            code_execution_enabled (bool): Whether to enable code execution
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.model_id = model_id
        self.search_enabled = search_enabled
        self.code_execution_enabled = code_execution_enabled

        if system_instruction is None:
            system_instruction = """
            You are a helpful assistant.
            """
            
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        
        try:
            self.chat = self.client.chats.create(
                model=model_id,
                config=config
            )
            return True
        except Exception as e:
            print(f"Error initializing chat: {e}")
            return False
    
    def send_message(self, message, pdf_file=None):
        """
        Send a message to the Gemini chat and get the response.
        If search is enabled, it will use search grounding.
        If code execution is enabled, it will use code execution.
        If pdf_file is provided, it will include the PDF in the message.
        
        Args:
            message (str): The message to send
            pdf_file (bytes, optional): PDF file content to include with the message
            
        Returns:
            str or dict: The response from Gemini (text or structured response with code execution)
        """
        if self.chat is None:
            raise ValueError("Chat not initialized. Call initialize_chat() first.")
            
        try:
            if pdf_file:
                # Handle PDF file processing directly through the model
                file_upload = self.client.files.upload(file=pdf_file)
                
                # Use model.generate_content directly for file processing
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[
                        file_upload,
                        message,
                    ]
                )
                return response.text
            
            elif self.search_enabled:
                # Use direct model generation with search tools for grounding
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=message,
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                google_search=types.GoogleSearch()
                            )
                        ]
                    )
                )
                return response.text
            
            elif self.code_execution_enabled:
                # Use direct model generation with code execution tool
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=message,
                    config=types.GenerateContentConfig(
                        tools=[
                            types.Tool(
                                code_execution=types.ToolCodeExecution()
                            )
                        ]
                    )
                )
                
                # Process the response to extract code execution parts
                result = {
                    "text": [],
                    "executable_code": [],
                    "code_execution_result": [],
                    "inline_data": []
                }
                
                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        result["text"].append(part.text)
                    if hasattr(part, 'executable_code') and part.executable_code is not None:
                        result["executable_code"].append(part.executable_code.code)
                    if hasattr(part, 'code_execution_result') and part.code_execution_result is not None:
                        result["code_execution_result"].append(part.code_execution_result.output)
                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                        result["inline_data"].append(part.inline_data.data)
                
                return result
            else:
                # Use normal chat without search grounding or code execution
                response = self.chat.send_message(message)
                return response.text
        except Exception as e:
            print(f"Error sending message: {e}")
            raise
    
    def process_files(self, files, query, file_types=None):
        """
        Process one or multiple files (PDF or CSV) with a specific query.
        
        Args:
            files: A single file or list of files to process (can be path, bytes, or list)
            query (str): The query about the file(s)
            file_types (list, optional): List of file types for each file. If None, will be determined from file extension.
            
        Returns:
            str: The text response from Gemini
        """
        try:
            # Handle single file case
            if not isinstance(files, list):
                files = [files]
                if file_types and not isinstance(file_types, list):
                    file_types = [file_types]
            
            # If file_types not provided, initialize as None for each file
            if not file_types:
                file_types = [None] * len(files)
            
            file_uploads = []
            for i, file in enumerate(files):
                # Determine file type if not specified
                file_type = file_types[i] if i < len(file_types) else None
                
                if not file_type:
                    if hasattr(file, 'name'):
                        # For uploaded files with name attribute
                        if file.name.lower().endswith('.csv'):
                            file_type = 'csv'
                        elif file.name.lower().endswith('.xlsx'):
                            file_type = 'excel'
                        elif file.name.lower().endswith('.pdf'):
                            file_type = 'pdf'
                        elif file.name.lower().endswith('.txt'):
                            file_type = 'txt'
                        elif file.name.lower().endswith(('.mp3', '.wav', '.m4a')):
                            file_type = 'audio'
                        elif file.name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                            file_type = 'video'
                        elif file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                            file_type = 'image'
                    elif isinstance(file, str):
                        # For file paths
                        if file.lower().endswith('.csv'):
                            file_type = 'csv'
                        elif file.lower().endswith('.xlsx'):
                            file_type = 'excel'
                        elif file.lower().endswith('.pdf'):
                            file_type = 'pdf'
                        elif file.lower().endswith('.txt'):
                            file_type = 'txt'
                        elif file.lower().endswith(('.mp3', '.wav', '.m4a')):
                            file_type = 'audio'
                        elif file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                            file_type = 'video'
                        elif file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                            file_type = 'image'
                
                # Process based on file type
                if file_type == 'csv':
                    # Process CSV file
                    file_upload = self._process_csv_file(file)
                elif file_type == 'excel':
                    # Process Excel file
                    file_upload = self._process_excel_file(file)
                elif file_type == 'txt':
                    # Process text file
                    file_upload = self._process_text_file(file)
                elif file_type == 'audio':
                    # Process audio file
                    file_upload = self._process_media_file(file, 'audio')
                elif file_type == 'video':
                    # Process video file
                    file_upload = self._process_media_file(file, 'video')
                elif file_type == 'image':
                    # Process image file
                    file_upload = self._process_image_file(file)
                else:
                    # Default to PDF processing
                    # If file is a string (path), convert to pathlib.Path
                    if isinstance(file, str):
                        file = pathlib.Path(file)
                    
                    # Upload the file
                    file_upload = self.client.files.upload(file=file)
                
                file_uploads.append(file_upload)
            
            # Generate content using the model with all files
            contents = file_uploads + [query]
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"Error processing files: {e}")
            raise
    
    def _process_csv_file(self, csv_file):
        """
        Process a CSV file and convert it to a format suitable for the Gemini API.
        
        Args:
            csv_file: CSV file to process (can be path, bytes, or file-like object)
            
        Returns:
            The processed file ready for the Gemini API
        """
        try:
            # Read the CSV file into a pandas DataFrame
            if isinstance(csv_file, str):
                # It's a file path
                df = pd.read_csv(csv_file)
            elif hasattr(csv_file, 'read'):
                # It's a file-like object (e.g., from st.file_uploader)
                df = pd.read_csv(csv_file)
            else:
                # It's bytes or another format
                df = pd.read_csv(io.BytesIO(csv_file))
            
            # Convert DataFrame to a formatted string
            csv_content = df.to_string(index=False)
            
            # Create a text representation that Gemini can process
            text_content = f"CSV Data:\n{csv_content}"
            
            # Upload as text content
            return {"text": text_content}
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            raise
    
    def _process_excel_file(self, excel_file):
        """
        Process an Excel (XLSX) file and convert it to a format suitable for the Gemini API.
        
        Args:
            excel_file: Excel file to process (can be path, bytes, or file-like object)
            
        Returns:
            The processed file ready for the Gemini API
        """
        try:
            # Read the Excel file into a pandas DataFrame
            if isinstance(excel_file, str):
                # It's a file path
                df = pd.read_excel(excel_file)
            elif hasattr(excel_file, 'read'):
                # It's a file-like object (e.g., from st.file_uploader)
                df = pd.read_excel(excel_file)
            else:
                # It's bytes or another format
                df = pd.read_excel(io.BytesIO(excel_file))
            
            # Convert DataFrame to a formatted string
            excel_content = df.to_string(index=False)
            
            # Create a text representation that Gemini can process
            text_content = f"Excel Data:\n{excel_content}"
            
            # Upload as text content
            return {"text": text_content}
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            raise
    
    def _process_text_file(self, text_file):
        """
        Process a text file and prepare it for the Gemini API.
        
        Args:
            text_file: Text file to process (can be path, bytes, or file-like object)
            
        Returns:
            The processed file ready for the Gemini API
        """
        try:
            # Read the text file content
            if isinstance(text_file, str):
                # It's a file path
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif hasattr(text_file, 'read'):
                # It's a file-like object (e.g., from st.file_uploader)
                text_content = text_file.read().decode('utf-8')
            else:
                # It's bytes or another format
                text_content = text_file.decode('utf-8')
            
            # Convert to pathlib.Path if it's a string path
            if isinstance(text_file, str):
                text_file = pathlib.Path(text_file)
                return self.client.files.upload(file=text_file)
            else:
                # For file-like objects or bytes, we need to create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
                    if isinstance(text_content, str):
                        tmp_file.write(text_content.encode('utf-8'))
                    else:
                        tmp_file.write(text_content)
                    tmp_path = pathlib.Path(tmp_file.name)
                    file_upload = self.client.files.upload(file=tmp_path)
                    # Clean up the temporary file
                    os.unlink(tmp_path)
                    return file_upload
        except Exception as e:
            print(f"Error processing text file: {e}")
            raise
    
    def _process_media_file(self, media_file, media_type):
        """
        Process an audio or video file and prepare it for the Gemini API.
        
        Args:
            media_file: Media file to process (can be path, bytes, or file-like object)
            media_type: Type of media ('audio' or 'video')
            
        Returns:
            The processed file ready for the Gemini API
        """
        try:
            # Convert to pathlib.Path if it's a string path
            if isinstance(media_file, str):
                media_file = pathlib.Path(media_file)
                file_upload = self.client.files.upload(file=media_file)
            elif hasattr(media_file, 'read'):
                # For file-like objects from streamlit uploader
                suffix = '.mp3' if media_type == 'audio' else '.mp4'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(media_file.getvalue())
                    tmp_path = pathlib.Path(tmp_file.name)
                    file_upload = self.client.files.upload(file=tmp_path)
                    # Clean up the temporary file
                    os.unlink(tmp_path)
            else:
                # It's bytes or another format
                suffix = '.mp3' if media_type == 'audio' else '.mp4'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(media_file)
                    tmp_path = pathlib.Path(tmp_file.name)
                    file_upload = self.client.files.upload(file=tmp_path)
                    # Clean up the temporary file
                    os.unlink(tmp_path)
            
            # For video files, we need to wait for processing to complete
            if media_type == 'video':
                while file_upload.state == "PROCESSING":
                    print('Waiting for video to be processed.')
                    time.sleep(5)
                    file_upload = self.client.files.get(name=file_upload.name)
                
                if file_upload.state == "FAILED":
                    raise ValueError(f"Video processing failed: {file_upload.state}")
                print(f'Video processing complete: {file_upload.uri}')
            
            return file_upload
        except Exception as e:
            print(f"Error processing {media_type} file: {e}")
            raise
    
    def process_pdf(self, pdf_files, query):
        """
        Process one or multiple PDF files with a specific query.
        This is maintained for backward compatibility.
        
        Args:
            pdf_files: A single PDF file or list of PDF files to process (can be path, bytes, or list)
            query (str): The query about the PDF(s)
            
        Returns:
            str: The text response from Gemini
        """
        return self.process_files(pdf_files, query, file_types=['pdf'] if isinstance(pdf_files, list) else 'pdf')

    def toggle_search(self, enabled=True):
        """
        Toggle search grounding on or off.
        If enabling search, code execution will be disabled.
        
        Args:
            enabled (bool): Whether to enable search grounding
            
        Returns:
            bool: The new state of search grounding
        """
        self.search_enabled = enabled
        if enabled:
            self.code_execution_enabled = False
        return self.search_enabled
    
    def toggle_code_execution(self, enabled=True):
        """
        Toggle code execution on or off.
        If enabling code execution, search will be disabled.
        
        Args:
            enabled (bool): Whether to enable code execution
            
        Returns:
            bool: The new state of code execution
        """
        self.code_execution_enabled = enabled
        if enabled:
            self.search_enabled = False
        return self.code_execution_enabled
    
    def is_search_enabled(self):
        """
        Check if search grounding is enabled.
        
        Returns:
            bool: True if search grounding is enabled, False otherwise
        """
        return self.search_enabled
        
    def is_code_execution_enabled(self):
        """
        Check if code execution is enabled.
        
        Returns:
            bool: True if code execution is enabled, False otherwise
        """
        return self.code_execution_enabled
    
    def is_initialized(self):
        """
        Check if the chat is initialized.
        
        Returns:
            bool: True if chat is initialized, False otherwise
        """
        return self.chat is not None
        
    def generate_images(self, prompt, number_of_images=1, person_generation="ALLOW_ADULT", aspect_ratio="1:1"):
        """
        Generate images using the Imagen model.
        
        Args:
            prompt (str): The text prompt to generate images from
            number_of_images (int): Number of images to generate (1-4)
            person_generation (str): Whether to allow person generation ('DONT_ALLOW' or 'ALLOW_ADULT')
            aspect_ratio (str): Aspect ratio of the generated images ('1:1', '3:4', '4:3', '16:9', '9:16')
            
        Returns:
            list: List of generated images as bytes
        """
        try:
            # Use the Imagen model for image generation
            result = self.client.models.generate_images(
                model="gemini-2.0-flash-preview-image-generation",
                prompt=prompt,
                config=dict(
                    number_of_images=number_of_images,
                    output_mime_type="image/jpeg",
                    person_generation=person_generation,
                    aspect_ratio=aspect_ratio
                )
            )
            
            # Convert GeneratedImage objects to bytes before returning
            image_bytes_list = []
            for image in result.generated_images:
                # Extract the bytes from each GeneratedImage object
                image_bytes_list.append(image.image.image_bytes)
                
            # Return the list of image bytes
            return image_bytes_list
        except Exception as e:
            print(f"Error generating images: {e}")
            raise

    def _process_image_file(self, image_file):
        """
        Process an image file and prepare it for the Gemini API.
        
        Args:
            image_file: Image file to process (can be path, bytes, or file-like object)
            
        Returns:
            The processed file ready for the Gemini API
        """
        try:
            # Convert to pathlib.Path if it's a string path
            if isinstance(image_file, str):
                image_file = pathlib.Path(image_file)
                file_upload = self.client.files.upload(file=image_file)
            elif hasattr(image_file, 'read'):
                # For file-like objects from streamlit uploader
                # Determine file extension from name if available
                suffix = '.jpg'  # Default extension
                if hasattr(image_file, 'name'):
                    _, ext = os.path.splitext(image_file.name)
                    if ext:
                        suffix = ext
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(image_file.getvalue())
                    tmp_path = pathlib.Path(tmp_file.name)
                    file_upload = self.client.files.upload(file=tmp_path)
                    # Clean up the temporary file
                    os.unlink(tmp_path)
            else:
                # It's bytes or another format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(image_file)
                    tmp_path = pathlib.Path(tmp_file.name)
                    file_upload = self.client.files.upload(file=tmp_path)
                    # Clean up the temporary file
                    os.unlink(tmp_path)
            
            return file_upload
        except Exception as e:
            print(f"Error processing image file: {e}")
            raise