# BT3017-Semester-2-AY25-26-Project: Graph Laplacian Visualiser
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

## Instructions
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
    
    6.Open the application in your browser and explore the Graph Laplacian Visualiser.
