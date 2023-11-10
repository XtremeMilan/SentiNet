# SentiNet

This project performs sentiment analysis on tweets using various libraries and a custom deep learning model. It includes a Streamlit web application for interactive sentiment analysis.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

### Prerequisites

Ensure you have the following dependencies installed:

- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/XtremeMilan/CPT-Project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd CPT-Project
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the Streamlit web application:

   ```bash
   streamlit run app.py
   ```

   The application will be accessible in your browser at [http://localhost:8501](http://localhost:8501).

2. Follow the instructions on the web application to input text and analyze sentiment using different libraries or the custom model.

## Project Structure

- `app.py`: Streamlit web application for sentiment analysis.
- `main.py`: Main script for loading the sentiment analysis model and running the application.
- `data/`: Directory containing the dataset.
- `checkpoints/`: Directory to save the trained model checkpoints.
- `requirements.txt`: List of Python dependencies for the project.

## Acknowledgments

- [Streamlit](https://www.streamlit.io/)
- [NLTK](https://www.nltk.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [AFINN](https://github.com/fnielsen/afinn)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://tfhub.dev/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/stable/)

## Contributing

Contributions are welcome! If you find any issues or have improvements to suggest, please open an issue or create a pull request.

## License

This project is open source and intended for educational purposes.
```

Feel free to modify the content further to better suit your project's specifics or to include additional information as needed.
