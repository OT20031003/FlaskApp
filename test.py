import google.generativeai as genai
import os

# --- API Key Configuration ---
# 1. Load from environment variable (Recommended)
#    Run `export GEMINI_API_KEY='YOUR_API_KEY'` in your terminal beforehand.
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please replace 'YOUR_API_KEY_HERE' in the code if hardcoding (not recommended).\n")
    
    # 2. Hardcode in the script (Not recommended: Security Risk)
    # API_KEY = 'YOUR_API_KEY_HERE'
    # if API_KEY == 'YOUR_API_KEY_HERE':
    #     print("--- !!! WARNING !!! ---")
    #     print("You have hardcoded the API key in the script.")
    #     print("For security reasons, using environment variables is strongly recommended.")
    #     print("----------------------\n")
    #     # Uncomment the following line to exit if needed
    #     # exit() 
    # genai.configure(api_key=API_KEY)


print("🤖 List of models available via Gemini API\n")

try:
    # Get the list of models
    models = genai.list_models()

    # Note: models is an iterator, so we can't check `if not models` directly easily 
    # without consuming it, but the loop handles empty lists gracefully.

    # Filter for models that support text generation (generateContent)
    generative_models = [
        m for m in models 
        if 'generateContent' in m.supported_generation_methods
    ]

    if not generative_models:
        print("No models found. Please check if your API key is correct.")

    print("--- Text Generation (generateContent) Supported Models ---")
    for m in generative_models:
        print(f"Model Name: {m.name}")
        print(f" Description: {m.description}")
        # print(f" Supported Methods: {m.supported_generation_methods}") # For details
        print("-" * 20)

    # (Optional) Models other than text generation (e.g., Embeddings)
    # Note: We need to call list_models() again or convert the first iterator to a list 
    # because the first iterator might be consumed. 
    # Here assuming we just iterate what's left or recall the API if needed.
    # For safety in this script structure, let's call list_models again to be sure 
    # or use the list logic if you prefer. 
    # *However, since 'models' variable above was consumed by list comprehension, 
    # we should fetch again or store in a list first.*
    
    models_refresh = genai.list_models()
    other_models = [
        m for m in models_refresh
        if 'generateContent' not in m.supported_generation_methods
    ]
    
    if other_models:
        print("\n--- Other Models (Embeddings, etc.) ---")
        for m in other_models:
            print(f"Model Name: {m.name}")
            # print(f" Supported Methods: {m.supported_generation_methods}") # For details
            print("-" * 20)


except Exception as e:
    print(f"An error occurred while retrieving models: {e}")
    print("Please check if your API key is set correctly and verify your network connection.")

# export GEMINI_API_KEY=""