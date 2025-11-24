import os
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

class ModelFactory:
    @staticmethod
    def get_model(model_name="llama3", temperature=0.7):
        """
        Returns a ChatModel instance.
        - If model_name starts with 'gemini', returns ChatGoogleGenerativeAI.
        - Otherwise, defaults to ChatOllama.
        """
        print(f"Initializing Model: {model_name} (Temp: {temperature})")
        
        if model_name.lower().startswith("gemini"):
            # Ensure GOOGLE_API_KEY is in your .env file
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables.")
                
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key,
                convert_system_message_to_human=True # Helps with some system prompt restrictions
            )
        else:
            return ChatOllama(
                model=model_name, 
                temperature=temperature
            )