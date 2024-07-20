# Customer_Service_chatbot-using-RAG

Customer Service Chatbot

This project is a Customer Service Chatbot powered by LangChain, Hugging Face, Pinecone, and Streamlit. It utilizes embeddings and language models to provide intelligent responses to user queries.

Project Setup

Prerequisites:
- Python 3.10 or later
- AWS EC2 (for deployment)
- Environment variables for API keys and other sensitive data

Dependencies:
Ensure you have the following packages installed:
- streamlit
- langchain
- langchain_community
- sentence-transformers
- pinecone-client
- python-dotenv
- huggingface_hub
- pyngrok
- pinecone-datasets

Installation:
1. Clone the repository:
   git clone <repository-url>
   cd <repository-directory>

2. Create a virtual environment:
   python -m venv myenv
   source myenv/bin/activate

3. Install the required packages:
   pip install -r requirements.txt

4. Set up your environment variables by creating a .env file in the root directory with the following content:
   PINECONE_API_KEY=<your-pinecone-api-key>
   HUGGINGFACE_API_KEY=<your-huggingface-api-key>

Running Locally:
To run the chatbot locally, use the following command:
   streamlit run app.py

Deploying on AWS EC2:
1. Launch an EC2 Instance:
   - Select an appropriate instance type with sufficient storage.
   - Configure security groups to allow access on port 8501 (Streamlit default).

2. Connect to Your EC2 Instance:
   ssh -i <your-key-pair>.pem ec2-user@<your-ec2-public-ip>

3. Install Dependencies:
   - Install Python and necessary packages.
   - Set up environment variables similarly to local setup.

4. Run the Application:
   streamlit run app.py

5. Access the Application:
   Open a web browser and go to http://<your-ec2-public-ip>:8501 to interact with your chatbot.

Troubleshooting:
- No Space Left on Device:
  Ensure you have sufficient disk space. Consider resizing your instance or attaching additional storage.

- Dependencies Not Installing:
  Verify that all dependencies are listed in requirements.txt and that your virtual environment is active.

Contributing:
Feel free to contribute to this project by submitting pull requests or opening issues.

License:
This project is licensed under the MIT License.
