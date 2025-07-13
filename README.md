# EduRAG: Intelligent Tutor Using RAG and LangChain

EduRAG is an intelligent tutor system designed to provide educational assistance by leveraging Retrieval-Augmented Generation (RAG) and LangChain. It allows users to upload educational content, ask questions, and receive AI-generated answers grounded in the provided knowledge base. It also supports natural language querying for database insights.

## Setup and Local Deployment

Follow these steps to get EduRAG running on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Axopher/EduRag_Assessment.git
cd edurag
```

### 2. Create a Virtual Environment and Install Dependencies

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a .env file in the root of your edurag directory and add your API keys.

```bash
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_MODEL_NAME="gemini-2.5-pro"
LLM_PROVIDER="gemini"
```

### 4. Run the FastAPI Application with Uvicorn

Open your terminal, navigate to the edurag directory, and start the FastAPI application:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Install Nginx

```bash
sudo nano /etc/nginx/sites-available/edurag.confsudo apt update
sudo apt install nginx
```

### 6. Configure Nginx as a Reverse Proxy:

Move that Nginx configuration file(i.e. edurag.conf) from your edurag directory to the given path

```bash
sudo mv edurag.conf /etc/nginx/sites-available/
```


### 7.Enable the Nginx Configuration:

Create a symbolic link from your edurag.conf file in sites-available to the sites-enabled directory. This tells Nginx to activate your new configuration.
```bash
sudo rm /etc/nginx/sites-enabled/default 
sudo ln -s /etc/nginx/sites-available/edurag.conf /etc/nginx/sites-enabled/
```

### 8. Restart Nginx:
Apply the new configuration by restarting the Nginx service:
```bash
sudo systemctl restart nginx
```

After these steps, ensure your FastAPI application is running with Uvicorn (as described in your README.md's "Run the FastAPI Application with Uvicorn" section), and then you should be able to access your EduRAG application by navigating to http://localhost/ in your web browser.