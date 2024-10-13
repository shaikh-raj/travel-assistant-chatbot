import os
import subprocess
import sys

def create_venv():
    if not os.path.exists('venv'):
        subprocess.run([sys.executable, '-m', 'venv', 'venv'])
    
    activate_script = os.path.join('venv', 'Scripts', 'activate')
        
    
    print(f"To activate the virtual environment, run:\nsource {activate_script}")

def create_requirements():
    requirements = [
        'pandas',
        'numpy',
        'transformers',
        'bertopic',
        'spacy',
        'tqdm',
        'streamlit'
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("Created requirements.txt")

def main():
    create_venv()
    create_requirements()
    print("\nNext steps:")
    print("1. Activate the virtual environment")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()