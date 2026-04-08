# BT3017-Semester-2-AY25-26-Project: Graph Laplacian Visualiser & AI Generated Video
This project demonstrates how to:

- build a graph with NetworkX
- compute the adjacency matrix, degree matrix, and Laplacian
- perform simple spectral clustering from Laplacian eigenvectors
- study range of influence using powers of the Laplacian

## Project files

- `Home.py`: Streamlit app for debugging and interaction
- `1_GraphSelection.py`: Choose the graph to visualise
- `2_Visualiser.py`: Visualise selected graph to showcase Laplacian properties
- `requirements.txt`: Python dependencies
- NotebookLM_AI_Generated_Topic_4_Graph_Data.mp4: AI Generated Graph Video

## Instructions for AI Generated Video
1. Download NotebookLM_AI_Generated_Topic_4_Graph_Data.mp4 from the github repository.

## Instructions for Graph Laplacian Visualiser
1. Clone the GitHub repository.
2. Create a Python virtual environment:
        ```
        python -m venv venv
        ```
3. Activate the virtual environment:
   
   Windows:
        ```
        venv\Scripts\activate
        ```

   macOS/Linux:
        ```
        source venv/bin/activate
        ```
4. Install the required dependencies:
        ```
        pip install -r requirements.txt
        ```
5. Run the Streamlit application:
        ```
        streamlit run Home.py
        ```
6. Open the application in your browser and explore the Graph Laplacian Visualiser.
