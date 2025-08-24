# CV Evaluation Project

## Project Introduction

This project aims to **automatically evaluate and analyze CVs (Curriculum Vitae)** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.

The idea is to provide a tool that helps HR teams or companies **quickly and accurately screen CVs** and classify them based on quality, experience, and relevant skills.

---

## Project Features

* Read CVs from **PDF** and **DOCX** files.
* Extract key information:

  * Name
  * Contact details (email, phone)
  * Education
  * Work experience
  * Skills
* Analyze CV quality using a **BERT** model for text evaluation.
* Provide an **automatic classification or score** for the CV.
* Simple interface to upload files and get results.

---

## Tools and Technologies Used

### 1. Programming Language

* **Python 3.13**: Main programming language for the project.

### 2. File Handling Libraries

* **PyPDF2**: For reading PDF files and extracting text.
* **python-docx**: For reading DOCX files.

### 3. Natural Language Processing

* **Transformers (Hugging Face)**: To use the **BERT** model for text analysis.
* **torch (PyTorch)**: To run the BERT model.
* **NLTK / spaCy**: For text cleaning, keyword extraction, and NLP preprocessing.

### 4. Machine Learning

* **PyTorch**: To build the CV classification model.
* **joblib / dill**: To save and load trained models.

### 5. Web API

* **FastAPI**: To create an API that handles CV uploads and processing.
* **CORS Middleware**: To allow requests from different browsers.
* **JSONResponse**: To return results in JSON format.

---

## How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run the local server**:

```bash
python api.py
```

* The API will be available at `http://127.0.0.1:8000`

3. **Upload a CV**:

* Through the web interface or using tools like Postman.
* Supports PDF or DOCX files.

4. **Receive results**:

* The system returns **JSON** containing:

  * Extracted key information
  * CV evaluation score
  * Important keywords

---

## Future Improvements

* Support multiple languages for CVs (English and Arabic).
* Enhance BERT model to better detect relevant skills.
* Create a full web interface with visual representation of results.
* Integrate a database to store CVs and their evaluations.

---

## Contributors

* **Shaimaa Dwedar** – Project Lead

---

## License

This project is open-source and can be used for educational and development purposes.



