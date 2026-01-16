
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
import datetime
from vertexai.preview import caching
from typing import Optional, List, Dict
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
PROJECT_ID = os.getenv("PROJECT_ID", "secret-tide-443909-k4")
LOCATION = os.getenv("LOCATION", "global")
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "../bigquery-service-key.json")
BUCKET_NAME = f"video-analysis-uploads-{PROJECT_ID}" # Dedicated bucket

# PostgreSQL Configuration (from .env file)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://pguser:NewSecurePassword2024@localhost:5432/video_analyzer_db")

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

# Initialize PostgreSQL connection pool
def get_db_connection():
    """Get a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return None

# Test database connection on startup
try:
    test_conn = get_db_connection()
    if test_conn:
        test_conn.close()
        logger.info("PostgreSQL database initialized successfully")
    else:
        logger.warning("PostgreSQL connection failed. Conversation history will not be persisted.")
except Exception as e:
    logger.warning(f"PostgreSQL connection failed: {str(e)}. Conversation history will not be persisted.")

def ensure_bucket_exists(bucket_name):
    """Creates the bucket if it doesn't exist."""
    try:
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            logger.info(f"Creating bucket {bucket_name}")
            bucket.create(location="us-central1") # Buckets can't be 'global', using a stable region
        return bucket
    except Exception as e:
        logger.error(f"Error checking/creating bucket: {e}")
        raise

async def upload_to_gcs(file: UploadFile, bucket_name: str) -> str:
    """Uploads a file to GCS and returns the gs:// URI."""
    try:
        bucket = ensure_bucket_exists(bucket_name)
        file_ext = os.path.splitext(file.filename)[1]
        blob_name = f"uploads/{uuid.uuid4()}{file_ext}"
        blob = bucket.blob(blob_name)
        blob.upload_from_file(file.file, content_type=file.content_type)
        return f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

def analyze_video_with_model(video_uri: str) -> dict:
    """
    1. Performs high-quality analysis.
    2. Creates a Context Cache for FAST future RAG chat.
    Uses gemini-3-pro-preview as it's the most stable/available in 'global'.
    """
    model_id = "gemini-3-pro-preview"
    video_part = Part.from_uri(uri=video_uri, mime_type="video/mp4")
    
    # 1. GENERATE ANALYSIS
    logger.info(f"Generating Analysis with {model_id}...")
    start_time = time.time()
    backend_prompt =  """
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

    ## 5. 🧠 Tactical Analysis (REFERENCE SPECIFIC VIDEO MOMENTS!)
    
    Analyze the tactics WITH specific timestamps and examples:
    
    **Player Strategies:**
    For each player, describe their tactical approach with video evidence:
    - Pattern: "Player X consistently does [behavior] - seen at [MM:SS], [MM:SS], [MM:SS]"
    - Intent: "This appears to be trying to [tactical goal]"
    - Effectiveness: "This worked/failed because [reason] - e.g., at [MM:SS] it resulted in [outcome]"
    
    **Control & Pressure Analysis:**
    - **Net Control**: "At [MM:SS] to [MM:SS], Player X dominated the T position, forcing Player Y to the back corners"
    - **Pressure Moments**: "At [MM:SS], Player Y was under extreme pressure due to [specific situation], resulting in [outcome]"
    
    **Wall Usage:**
    - Describe specific shots: "At [MM:SS], Player X used the side wall for a [shot type], which [outcome]"
    - Missed opportunities: "At [MM:SS], Player Y could have used the back wall but chose [alternative] instead"
    
    **Shot Selection Examples:**
    - Good: "At [MM:SS], Player X's choice to [shot type] was excellent because [tactical reason]"
    - Poor: "At [MM:SS], Player Y's [shot type] was a mistake because [tactical reason]"
    
    ✓ Every tactical point must have video timestamp
    ✓ Explain cause and effect with specific examples
    ✗ NO generic statements like "Player X has good strategy"

    ---

    ## 6. 🏋️ Performance Weaknesses (MUST BE SPECIFIC TO THIS VIDEO!)
    
    For EACH player, identify 5-7 specific weaknesses WITH timestamps:
    
    Format for each weakness:
    - **[MM:SS] - Weakness Name**: Describe exactly what happened in the video and why it was a weakness
    
    Example:
    **Player 1 Weaknesses:**
    - **[00:15] - Poor Recovery Speed**: After hitting a forehand from the back right, Player 1 walked back to the T instead of running, arriving late and leaving the front court exposed
    - **[00:32] - Predictable Shot Selection**: Hit 4 consecutive cross-court drives, allowing Player 2 to anticipate and intercept at the volley
    - **[00:47] - Weak Return of Serve**: Popped up a short return that landed mid-court, giving Player 2 an easy attacking opportunity
    - **[01:03] - Late Split-Step**: Was still moving when Player 2 struck the ball, couldn't react to the drop shot
    - **[01:18] - Racket Preparation**: Started backswing after the ball bounced, resulting in a rushed, defensive shot
    
    ✓ Must include timestamps
    ✓ Must describe actual video moments
    ✓ Must explain why it's a weakness
    ✗ NO generic weaknesses without video evidence

    ---

    ## 7. 🚀 Improvement Recommendations (BASED ON VIDEO EVIDENCE!)
    
    Provide 5-7 SPECIFIC improvement areas with:
    1. What you observed in the video
    2. Specific drill or practice method
    3. Expected outcome
    
    Format:
    **Improvement Area #N: [Specific Issue Observed]**
    - **Video Evidence**: At [MM:SS], saw [specific behavior]
    - **Drill to Fix**: [Specific drill name and how to do it]
    - **Focus Point**: [One key thing to concentrate on]
    - **Expected Result**: [How this will improve performance]
    
    Example:
    **Improvement Area #1: Recovery to T Position**
    - **Video Evidence**: At 00:23, 00:45, and 01:12, players were caught out of position because they didn't return to the T
    - **Drill to Fix**: "Ghost ball T-recovery" - hit imaginary shot from corner, sprint to T, touch center with racket, repeat from opposite corner
    - **Focus Point**: First step must be towards the T, not towards the next anticipated shot
    - **Expected Result**: Will be in position for 80% more shots instead of being stretched
    
    ✓ Reference specific video moments
    ✓ Name actual drills (not just "practice more")
    ✓ Explain the connection between video observation and drill

    ---

    ## 8. 🎯 Player-Specific Recommendations (MANDATORY - MUST BE VIDEO-SPECIFIC!)
    
    **CRITICAL: DO NOT give generic advice. Every recommendation MUST reference specific moments from THIS video.**
    
    For EACH player, provide 5-7 recommendations in this EXACT format:
    
    **Player X - Recommendation #N:**
    - **At [MM:SS]**: Describe what the player actually did in the video
    - **Should have done**: Specific alternative action they should have taken instead
    - **Why**: Explain the tactical/technical benefit of doing it differently
    - **How to practice**: One specific drill or focus point to improve this
    
    Example format:
    **Player 1 - Recommendation #1:**
    - **At [00:23]**: Player 1 hit a cross-court drive while standing at mid-court
    - **Should have done**: Hit a straight drive down the left wall to push Player 2 deep
    - **Why**: Cross-court at mid-court gave Player 2 an easy volley opportunity, leading to the point loss. Straight drives from mid-court are safer and maintain pressure
    - **How to practice**: Practice "mid-court decision making" drill - only hit cross-court when at the front of the court
    
    **Player 1 - Recommendation #2:**
    - **At [00:45]**: Player 1 stayed at the back after returning a lob
    - **Should have done**: Immediately moved to the T position after hitting the return
    - **Why**: Staying back left Player 1 out of position when Player 2 hit a drop shot, making it unreachable
    - **How to practice**: "Return and recover" drill - hit from back corner, then sprint to T
    
    REQUIREMENTS:
    ✓ Must reference actual timestamps from the video
    ✓ Must describe what actually happened in the video
    ✓ Must give specific alternative actions
    ✓ Must explain tactical reasoning
    ✓ Must provide practical drill/practice method
    ✗ NO generic advice like "improve footwork" or "work on positioning"
    ✗ NO advice that could apply to any video
    
    Provide 5-7 recommendations per player minimum.

    ---

    ## 9. 🎭 Psychological & Match Insights
    Describe:
    - Confidence level
    - Pressure handling
    - Momentum shifts
    - Body language & mental strength

    ---

    ## Output Rules
    - **CRITICAL**: Every recommendation, weakness, and insight MUST reference specific timestamps [MM:SS]
    - **CRITICAL**: Describe what actually happened in THIS video, not generic coaching advice
    - **CRITICAL**: If you say "Player X should have done Y", explain what they actually did instead at a specific moment
    - Be concise but highly informative  
    - Use bullet points and tables where helpful  
    - No hallucinations — only describe what is actually visible in the video
    - Prioritize video-specific analysis over generic coaching advice
    - Think: "A coach reviewing this specific video" NOT "A coaching textbook"
    
    **Quality Check:**
    ❌ BAD: "Player 1 needs to improve footwork" (too generic)
    ✅ GOOD: "At [00:34], Player 1's footwork was too narrow on the backhand, causing them to reach instead of step, resulting in a weak return that Player 2 volleyed for a winner"
    
    ❌ BAD: "Work on positioning" (could apply to any video)
    ✅ GOOD: "At [00:52], Player 1 stood too close to the front wall after their drop shot, blocking their own retreat path when Player 2 hit a lob. Should have moved to the T immediately after the drop"  

    """
    
    try:
        model = GenerativeModel(model_id)
        response = model.generate_content(
            [video_part, backend_prompt],
            generation_config={"max_output_tokens": 4096, "temperature": 0.2}
        )
        analysis_text = response.text
        logger.info(f"Analysis generated in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI Analysis failed: {str(e)}")

    # 2. CREATE CACHE for RAG
    logger.info("Creating context cache for Chat...")
    cache_start = time.time()
    try:
        cache = caching.CachedContent.create(
            model_name=model_id,
            system_instruction="You are a helpful expert video assistant specialized in squash and padel. You can answer questions about the video analysis, player performance, tactical decisions, and provide detailed if-then recommendations based on what you observe in the footage.",
            contents=[video_part],
            ttl=datetime.timedelta(minutes=60),
        )
        cache_name = cache.name
        logger.info(f"Cache created: {cache_name} in {time.time() - cache_start:.2f} seconds")
    except Exception as e:
        logger.warning(f"Caching not supported in this region/project: {e}")
        cache_name = None
    
    return {
        "analysis": analysis_text,
        "cache_name": cache_name
    }

class UrlRequest(BaseModel):
    url: str

@app.post("/analyze_url")
async def analyze_url(request: UrlRequest):
    logger.info(f"Processing URL: {request.url}")
    start_time = time.time()
    try:
        temp_filename = f"temp_{uuid.uuid4()}.mp4"
        # Updated command with cookies and better handling for YouTube bot detection
        command = [
            "yt-dlp",
            "-f", "best[ext=mp4]/best",
            "-o", temp_filename,
            "--force-overwrites",
            "--cookies-from-browser", "chrome",  # Try to use Chrome cookies
            "--no-check-certificates",
            request.url
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"Video downloaded successfully: {temp_filename}")
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed: {e.stderr}")
        # Try without cookies if Chrome cookies fail
        try:
            logger.info("Retrying without browser cookies...")
            command = [
                "yt-dlp",
                "-f", "best[ext=mp4]/best",
                "-o", temp_filename,
                "--force-overwrites",
                "--no-check-certificates",
                "--extractor-args", "youtube:player_client=android",
                request.url
            ]
            subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"Video downloaded successfully on retry: {temp_filename}")
        except Exception as retry_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Download failed: YouTube requires authentication. Please try: 1) Using a public/unlisted video, 2) Upload the video file directly instead"
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

    try:
        video_uri = await upload_to_gcs_from_path(temp_filename, BUCKET_NAME)
        os.remove(temp_filename)
    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    result = analyze_video_with_model(video_uri)
    return {**result, "video_uri": video_uri, "processing_time": time.time() - start_time}

async def upload_to_gcs_from_path(file_path: str, bucket_name: str) -> str:
    bucket = ensure_bucket_exists(bucket_name)
    blob_name = f"uploads/{uuid.uuid4()}.mp4"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{bucket_name}/{blob_name}"

class ChatRequest(BaseModel):
    query: str
    video_uri: Optional[str] = None
    cache_name: Optional[str] = None
    session_id: Optional[str] = None

def store_conversation_in_db(session_id: str, role: str, content: str):
    """Store a conversation message in PostgreSQL database."""
    conn = get_db_connection()
    if not conn:
        logger.warning("No database connection available")
        return
    
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO conversation_history (session_id, role, content, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (session_id, role, content, datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Stored message for session {session_id}")
    except Exception as e:
        logger.warning(f"Failed to store conversation in database: {e}")
        if conn:
            conn.close()

def get_conversation_history(session_id: str) -> List[Dict]:
    """Retrieve conversation history from PostgreSQL database."""
    conn = get_db_connection()
    if not conn:
        logger.warning("No database connection available")
        return []
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT role, content, EXTRACT(EPOCH FROM timestamp) as timestamp
            FROM conversation_history
            WHERE session_id = %s
            ORDER BY timestamp ASC
        """
        cursor.execute(query, (session_id,))
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Convert RealDictRow to regular dict
        return [dict(msg) for msg in messages]
    except Exception as e:
        logger.warning(f"Failed to retrieve conversation history: {e}")
        if conn:
            conn.close()
        return []

@app.post("/rag_chat")
async def rag_chat(request: ChatRequest):
    start_time = time.time()
    try:
        logger.info(f"Received Chat Request: session_id={request.session_id}, cache_name={request.cache_name}, video_uri={'provided' if request.video_uri else 'None'}")
        logger.info(f"Chat query: {request.query}")
        model_id = "gemini-3-pro-preview"
        
        # Get conversation history from Redis if session_id provided
        conversation_context = ""
        if request.session_id:
            history = get_conversation_history(request.session_id)
            if history:
                conversation_context = "\n\nPrevious conversation context:\n"
                for msg in history[-10:]:  # Last 10 messages for context
                    role = "User" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"
                conversation_context += "\nCurrent question:\n"
        
        # Build the full query with conversation context
        full_query = conversation_context + request.query if conversation_context else request.query
        
        if request.cache_name:
            logger.info(f"Fetching cache: {request.cache_name}")
            cache_fetch_start = time.time()
            cache = caching.CachedContent.get(request.cache_name)
            logger.info(f"Cache fetched in {time.time() - cache_fetch_start:.2f} seconds")
            
            model = GenerativeModel.from_cached_content(cached_content=cache)
            
            logger.info("Generating content using CACHE...")
            gen_start = time.time()
            response = model.generate_content(full_query)
            logger.info(f"Content generated via CACHE in {time.time() - gen_start:.2f} seconds")
        else:
            if not request.video_uri:
                raise HTTPException(status_code=400, detail="Missing video context")
            
            logger.info("Using standard mode (NO CACHE)...")
            model = GenerativeModel(model_id)
            video_part = Part.from_uri(uri=request.video_uri, mime_type="video/mp4")
            
            logger.info("Generating content via SLOW mode...")
            gen_start = time.time()
            response = model.generate_content([video_part, full_query])
            logger.info(f"Content generated via SLOW mode in {time.time() - gen_start:.2f} seconds")
        
        # Store conversation in PostgreSQL database
        if request.session_id:
            store_conversation_in_db(request.session_id, "user", request.query)
            store_conversation_in_db(request.session_id, "assistant", response.text)
        
        total_time = time.time() - start_time
        logger.info(f"Total RAG Chat request took {total_time:.2f} seconds")
        return {"answer": response.text}
    except Exception as e:
        logger.error(f"Chat failed after {time.time() - start_time:.2f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_and_analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    start_time = time.time()
    video_uri = await upload_to_gcs(file, BUCKET_NAME)
    result = analyze_video_with_model(video_uri)
    return {**result, "video_uri": video_uri, "processing_time": time.time() - start_time}

@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history for a given session."""
    try:
        history = get_conversation_history(session_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to retrieve conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
