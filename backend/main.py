
import os
import time
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import subprocess
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PROJECT_ID = "secret-tide-443909-k4"
LOCATION = "global"
KEY_PATH = "../bigquery-service-key.json"
BUCKET_NAME = f"video-analysis-uploads-{PROJECT_ID}" # Dedicated bucket

# Initialize Vertex AI & Storage
try:
    if os.path.exists(KEY_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        storage_client = storage.Client()
        logger.info("Vertex AI & Storage initialized successfully")
    else:
        logger.error(f"Key file not found at {KEY_PATH}")
except Exception as e:
    logger.error(f"Failed to initialize GCC services: {str(e)}")

def ensure_bucket_exists(bucket_name):
    """Creates the bucket if it doesn't exist."""
    try:
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            logger.info(f"Creating bucket {bucket_name}")
            bucket.create(location=LOCATION)
        return bucket
    except Exception as e:
        logger.error(f"Error checking/creating bucket: {e}")
        raise

async def upload_to_gcs(file: UploadFile, bucket_name: str) -> str:
    """Uploads a file to GCS and returns the gs:// URI."""
    try:
        bucket = ensure_bucket_exists(bucket_name)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        blob_name = f"uploads/{uuid.uuid4()}{file_ext}"
        blob = bucket.blob(blob_name)
        
        # Stream upload
        # Warning: For very large files, this loads into memory if we read() all at once.
        # But requests are often streamed. We'll use upload_from_file.
        blob.upload_from_file(file.file, content_type=file.content_type)
        
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

def analyze_video_with_model(video_uri: str) -> dict:
    """Core analysis logic reusable for both upload and URL."""
    # Fallback logic for model selection
    target_model_id = "gemini-3-pro-preview"
    try:
            # Try instantiating 3.0 
        model = GenerativeModel(target_model_id)
    except:
        logger.warning(f"Model {target_model_id} not found, falling back to 2.5 Pro")
        model = GenerativeModel("gemini-1.5-pro-002") # keeping 1.5 as safe fallback if 2.5 not avail in this env

    # Backend Managed Prompt for "Analysis and RAG"
    backend_prompt = """
    You are an elite **Padel & Squash Video Performance Analysis AI** used by professional coaches and sports analysts.

    The user uploads a match or training video.  
    Your job is to deeply analyze player movement, shot quality, positioning, tactics, and match dynamics.

    Analyze the video and return the following in **clean structured Markdown**:

    ---

    ## 1. 🎯 Match Summary
    Provide a clear and professional summary of:
    - Type of clip (match, rally, drill, training)
    - Number of players and teams
    - Overall flow of play
    - Who is dominant and why
    - Key turning points in the clip

    ---

    ## 2. 🧍 Players & Visual Entities
    Identify all visible elements:
    - Players (Player 1, Player 2, etc.)
    - Court type (padel or squash)
    - Racket, ball, walls, glass, service boxes
    - On-screen text, scoreboards, timers (if visible)

    ---

    ## 3. ⏱️ Second-by-Second Timeline (RAG-ready)
    Break the clip into a **timestamped timeline**:

    Format:
    [00:00 – 00:01] Player 1 serves from right side
    [00:02 – 00:03] Player 2 returns cross-court
    [00:04 – 00:05] Rally shifts to back-court


    Each entry should include:
    - Who hits the ball
    - Shot type (drive, lob, volley, smash, boast, drop, etc.)
    - Court position (front, mid-court, back-court)
    - Rally momentum changes

    ---

    ## 4. 📊 Performance Metrics (Estimate visually)
    Derive **approximate metrics** from the video:
    - Rally length (avg & max)
    - Shot distribution (% volleys, smashes, lobs, drives)
    - Court positioning time (% at net vs back-court)
    - Aggression level (defensive / neutral / attacking)
    - Movement intensity (low / medium / high)

    ---

    ## 5. 🧠 Tactical Analysis
    Explain:
    - What strategy each player/team is using
    - Who controls the net and who is under pressure
    - Whether the players are using walls effectively
    - Shot selection quality

    ---

    ## 6. 🏋️ Performance Weaknesses
    List specific, coach-style weaknesses:
    - Footwork issues
    - Late positioning
    - Poor shot selection
    - Weak returns
    - Over-reliance on defense
    - Lack of net control

    ---

    ## 7. 🚀 Improvement Recommendations
    Give **actionable coaching advice**, for example:
    - What drills they should practice
    - How to improve positioning
    - How to convert defense to attack
    - How to win more points

    ---

    ## 8. 🎭 Psychological & Match Insights
    Describe:
    - Confidence level
    - Pressure handling
    - Momentum shifts
    - Body language & mental strength

    ---

    ## Output Rules
    - Be concise but highly informative  
    - Use bullet points and tables where helpful  
    - No hallucinations — only infer what is visible  
    - Prioritize sports performance over generic video description  

    """

    video_part = Part.from_uri(uri=video_uri, mime_type="video/mp4")

    response = model.generate_content(
        [video_part, backend_prompt],
        generation_config={
            "max_output_tokens": 4096,
            "temperature": 0.2, # Lower temp for more factual analysis
        }
    )
    
    return response.text

class UrlRequest(BaseModel):
    url: str

@app.post("/analyze_url")
async def analyze_url(request: UrlRequest):
    """
    Analyzes a video from a URL (YouTube, direct MP4, etc.).
    1. Downloads video locally using yt-dlp.
    2. Uploads to GCS.
    3. Analyzes.
    """
    logger.info(f"Processing URL: {request.url}")
    start_time = time.time()
    
    # 1. Download Video
    try:
        # Create temp filename
        temp_filename = f"temp_{uuid.uuid4()}.mp4"
        
        # yt-dlp command to download strictly as mp4
        # We limit specific qualities to avoid massive 4k files for this POC
        # CHANGED: simplified format to avoid 'merging' which requires ffmpeg
        command = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", temp_filename,
            "--force-overwrites",
            request.url
        ]
        
        logger.info("Downloading video...")
        subprocess.run(command, check=True)
        
        if not os.path.exists(temp_filename):
             raise Exception("Download failed, file not found")
             
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download video from URL: {str(e)}")

    # 2. Upload to GCS
    try:
        bucket = ensure_bucket_exists(BUCKET_NAME)
        blob_name = f"uploads/url_sourced_{uuid.uuid4()}.mp4"
        blob = bucket.blob(blob_name)
        
        logger.info(f"Uploading {temp_filename} to {blob_name}")
        blob.upload_from_filename(temp_filename)
        
        video_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        
        # Cleanup local file
        os.remove(temp_filename)
        
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video to cloud storage: {str(e)}")

    # 3. Analyze
    try:
        analysis_text = analyze_video_with_model(video_uri)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "analysis": analysis_text,
            "video_uri": video_uri,
            "processing_time": duration
        }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class ChatRequest(BaseModel):
    query: str
    video_uri: str

@app.post("/rag_chat")
async def rag_chat(request: ChatRequest):
    """
    RAG-style chat with the video.
    The video is already in GCS. We send the video Part + User Query to Gemini.
    Gemini's 1M+ context window handles this natively (Long Context RAG).
    """
    try:
        logger.info(f"Chat query: {request.query} for video: {request.video_uri}")
        
        # Instantiate Model
        target_model_id = "gemini-3-pro-preview"
        try:
             model = GenerativeModel(target_model_id)
        except:
             model = GenerativeModel("gemini-1.5-pro-002")

        # Create the prompt
        # We instruct the model to act as a specific expert based on the video
        chat_prompt = f"""
        You are a helpful expert video assistant. 
        Answer the user's question based strictly on the visible and audible content of the video provided.
        Timestamp your answers where possible (e.g. "At 02:15...").
        
        User Question: {request.query}
        """

        video_part = Part.from_uri(uri=request.video_uri, mime_type="video/mp4")

        # Generate answer
        # Note: For a real conversational history, we would send the history list here.
        # For this POC, we treat each question as a single-turn RAG query against the video.
        response = model.generate_content(
            [video_part, chat_prompt],
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.3,
            }
        )
        
        return {"answer": response.text}

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/upload_and_analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Legacy endpoint refactored to use helper.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    logger.info(f"Processing upload: {file.filename}")
    
    start_time = time.time()
    
    # 1. Upload
    video_uri = await upload_to_gcs(file, BUCKET_NAME)
    logger.info(f"Video uploaded to: {video_uri}")

    # 2. Analyze
    try:
        analysis_text = analyze_video_with_model(video_uri)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "analysis": analysis_text,
            "video_uri": video_uri,
            "processing_time": duration
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Allow running the script directly
    uvicorn.run(app, host="0.0.0.0", port=8000)
