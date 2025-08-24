from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import os
import tempfile
from docx import Document
from PyPDF2 import PdfReader
import re
import numpy as np
from typing import Dict, Any
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI + CORS
app = FastAPI(
    title="CV Evaluation API",
    description="API for evaluating CVs using deep learning",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:8000",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and resources
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    label_encoder = joblib.load("label_encoder.pkl")
    specialization_encoder = joblib.load("specialization_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    logger.info("All resources loaded successfully")
except Exception as e:
    logger.error(f"Error loading resources: {e}")
    raise RuntimeError("Failed to load required resources")

class CVScoringModel(nn.Module):
    def __init__(self, num_specs):
        super(CVScoringModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.spec_embedding = nn.Embedding(num_embeddings=num_specs, embedding_dim=16)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size + 2 + 16, 3)

    def forward(self, input_ids, attention_mask, features, spec):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        spec_embed = self.spec_embedding(spec)
        combined = torch.cat((pooled_output, features, spec_embed), dim=1)
        logits = self.fc(self.dropout(combined))
        return logits

try:
    model = CVScoringModel(num_specs=len(specialization_encoder.classes_))
    model.load_state_dict(torch.load("cv_scoring_classifier.pt", map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model")

# Helper functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Unable to extract text from PDF file")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file"""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=400, detail="Unable to extract text from DOCX file")

def extract_experience(cv_text: str) -> int:
    """Extract years of experience from text"""
    exp_patterns = [
        r'(\d+)\s*(?:years?|yrs?)',
        r'experience\s*[:\\-]\s*(\d+)',
    ]
    
    experience_years = 0
    for pattern in exp_patterns:
        matches = re.findall(pattern, cv_text, re.IGNORECASE)
        if matches:
            try:
                exp_values = [int(m) for m in matches if str(m).isdigit()]
                if exp_values:
                    experience_years = max(exp_values)
                    break
            except ValueError:
                continue
    
    return min(experience_years, 50)  

def extract_specialization(cv_text: str) -> str:
    """Extract specialization from text"""
    specializations = specialization_encoder.classes_
    cv_text_lower = cv_text.lower()
    
    specialization = "General"
    max_occurrences = 0
    
    for spec in specializations:
        spec_lower = spec.lower()
        pattern = r'\b' + re.escape(spec_lower) + r'\b'
        matches = re.findall(pattern, cv_text_lower)
        
        if matches and len(matches) > max_occurrences:
            max_occurrences = len(matches)
            specialization = spec
    
    return specialization

def extract_features(cv_text: str) -> Dict[str, Any]:
    """Extract all features from CV text"""
    cv_text_clean = re.sub(r'\s+', ' ', cv_text).strip()
    
    words = cv_text_clean.split()
    sentences = re.split(r'[.!?]+', cv_text_clean)
    
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)
    
    experience_years = extract_experience(cv_text)
    specialization = extract_specialization(cv_text)
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "experience_years": experience_years,
        "specialization": specialization,
        "text_length": len(cv_text)
    }

# Endpoints
@app.get("/")
async def root():
    """Homepage"""
    return {
        "message": "Welcome to the CV Evaluation System",
        "version": "1.0.0",
        "endpoints": {
            "/evaluate_cv": "Evaluate a CV (POST)",
            "/health": "Check server health",
            "/model_info": "Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Check server health"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": True,
        "resources_loaded": True
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "CVScoringModel",
        "num_specializations": len(specialization_encoder.classes_),
        "labels": label_encoder.classes_.tolist(),
        "specializations": specialization_encoder.classes_.tolist(),
        "device": str(device)
    }

@app.post("/evaluate_cv")
async def evaluate_cv(
    age: int = Form(..., description="Age of the CV owner", ge=18, le=70),
    file: UploadFile = File(..., description="CV file (PDF or DOCX)")
):
    """
    Evaluate a CV and estimate the level of the applicant
    """
    try:
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(
                status_code=400,
                detail="File must be PDF or DOCX"
            )
        
        if age < 18 or age > 70:
            raise HTTPException(
                status_code=400,
                detail="Age must be between 18 and 70"
            )

        suffix = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if suffix == '.pdf':
                cv_text = extract_text_from_pdf(tmp_path)
            elif suffix == '.docx':
                cv_text = extract_text_from_docx(tmp_path)
            
            if not cv_text or len(cv_text.strip()) < 100:
                raise HTTPException(
                    status_code=400,
                    detail="Unable to extract sufficient text from file. It may be empty or protected"
                )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        features_dict = extract_features(cv_text)
        
        try:
            spec_encoded = specialization_encoder.transform([features_dict["specialization"]])[0]
            features_scaled = scaler.transform([[age, features_dict["experience_years"]]])[0]
        except Exception as e:
            logger.error(f"Feature encoding error: {e}")
            raise HTTPException(status_code=500, detail="Data processing error")

        encoding = tokenizer(
            cv_text,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float).unsqueeze(0).to(device)
        spec_tensor = torch.tensor([spec_encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, features_tensor, spec_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_label = label_encoder.inverse_transform([predicted_idx.item()])[0]
        confidence_value = confidence.item()

        all_scores = {
            label: round(probabilities[0][i].item() * 100, 2)
            for i, label in enumerate(label_encoder.classes_)
        }

        response = {
            "success": True,
            "data": {
                "personal_info": {
                    "age": age,
                    "estimated_experience": features_dict["experience_years"]
                },
                "cv_analysis": {
                    "word_count": features_dict["word_count"],
                    "sentence_count": features_dict["sentence_count"],
                    "avg_sentence_length": features_dict["avg_sentence_length"],
                    "specialization": features_dict["specialization"],
                    "text_length": features_dict["text_length"]
                },
                "evaluation": {
                    "level": predicted_label,
                    "confidence": round(confidence_value * 100, 2),
                    "score_breakdown": all_scores,
                    "recommendation": get_recommendation(predicted_label, features_dict)
                }
            }
        }

        logger.info(f"CV evaluated successfully. Level: {predicted_label}")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error during request processing")

def get_recommendation(level: str, features: Dict[str, Any]) -> str:
    """Generate recommendations based on CV level"""
    recommendations = {
        "High": "Excellent CV! Recommended to apply for senior positions.",
        "Medium": "Good CV but needs some improvements in structure and content.",
        "Low": "Consider rewriting the CV and improving content and skills."
    }
    
    additional_tips = []
    
    if features["word_count"] < 200:
        additional_tips.append("The CV is too short; add more details about your experience and skills.")
    elif features["word_count"] > 1000:
        additional_tips.append("The CV is too long; try to condense content to key points.")
    
    if features["avg_sentence_length"] > 25:
        additional_tips.append("Sentences are too long; try splitting them into shorter, clearer sentences.")
    
    main_recommendation = recommendations.get(level, "Consider reviewing the CV with a specialist.")
    
    if additional_tips:
        return main_recommendation + " " + " ".join(additional_tips)
    
    return main_recommendation

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
