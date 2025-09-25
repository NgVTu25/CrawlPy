"""
IELTS Speaking Dataset Collection Scripts
Bộ script thu thập và xử lý dữ liệu cho IELTS Speaking
"""

import os
import csv
import json
import time
import asyncio
import re
import warnings
import logging
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ['absl', 'grpc', 'google.auth.transport.grpc']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Configure GRPC environment
os.environ.update({
    'GRPC_VERBOSITY': 'none',
    'GRPC_TRACE': 'none',
    'GRPC_ENABLE_FORK_SUPPORT': '0',
    'GRPC_DNS_RESOLVER': 'native',
    'GOOGLE_CLOUD_SUPPRESS_RUBY_WARNINGS': 'true',
})

# Disable ALTS credentials warning
def dummy_callback(*args, **kwargs):
    pass

try:
    import grpc
    grpc.alts_channel_credentials = dummy_callback
except ImportError:
    pass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import json

# Required libraries (install via pip)
# pip install yt-dlp whisper pydub pandas tqdm youtube-transcript-api google-generativeai

from sympy import true
import yt_dlp
import whisper
import pandas as pd
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
# from pydub import AudioSegment  # Requires additional setup
import google.generativeai as genai

# ================== Configuration ==================
class Config:
    # Directories
    BASE_DIR = Path("IELTS_Speaking_Dataset")
    SCORING_DIR = BASE_DIR / "scoring"
    PRACTICE_DIR = BASE_DIR / "practice"
    
    # Audio settings
    AUDIO_FORMAT = "wav"
    SAMPLE_RATE = 16000
    
    # API Keys - List of backup keys
    API_KEYS = [
        "",  # Primary key
        "",  # Backup key 1
        "",  # Backup key 2
        "",  # Backup key 3
    ]
    
    # API Rate Limits and Batch Settings
    MAX_GEMINI_REQUESTS_PER_MINUTE = 60
    GEMINI_BATCH_SIZE = 10
    GEMINI_REQUEST_DELAY = 1.0  # seconds between requests
    GEMINI_BATCH_DELAY = 2.0    # seconds between batches
    
    _current_key_index = 0
    
    @classmethod
    def get_api_key(cls):
        """Get current API key"""
        return cls.API_KEYS[cls._current_key_index]
    
    @classmethod
    def rotate_api_key(cls):
        """Rotate to next API key"""
        cls._current_key_index = (cls._current_key_index + 1) % len(cls.API_KEYS)
        return cls.get_api_key()
    GEMINI_API_KEY = "INPUT YOU API"  
    
    # YouTube search queries
    YOUTUBE_QUERIES = [
        "IELTS Speaking Band 6.0 test real exam",
        "IELTS Speaking Band 7.0 test with examiner",
        "IELTS Speaking Band 8.0 sample answer",
        "IELTS Speaking Part 1 2 3 full test",
        "Cambridge IELTS Speaking test recording"
    ]
    
    # IELTS Topics
    IELTS_TOPICS = [
        "Hometown", "Work/Study", "Technology", "Environment",
        "Education", "Health", "Food", "Travel", "Books",
        "Music", "Sports", "Shopping", "Friends", "Family"
    ]

# ================== Data Models ==================
@dataclass
class ScoringData:
    id: str
    part: int
    topic: str
    question: str
    answer: str
    transcript: str
    band_overall: float
    fluency: float
    grammar: float
    lexical: float
    pronunciation: float
    feedback: str
    audio_path: str
    
@dataclass
class PracticeData:
    id: str
    part: int
    question: str
    band: float
    sample_answer: str
    topic: str
    keywords: str

# ================== Dataset A: Scoring Dataset ==================

class ScoringDataCollector:
    def __init__(self):
        self.setup_directories()
        self.whisper_model = None
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(Config.SCORING_DIR / "audio", exist_ok=True)
        os.makedirs(Config.SCORING_DIR / "transcripts", exist_ok=True)
        
    def download_youtube_videos(self, max_videos: int = 10) -> List[Dict]:
        """Download YouTube videos for IELTS Speaking tests"""
        print("🔍 Searching and downloading YouTube videos...")
        
        # First, let's check for ffmpeg
        print("⚙️ Checking for ffmpeg...")
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        
        if not ffmpeg_path:
            print("⚠️ ffmpeg not found. Please install ffmpeg first:")
            print("1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases")
            print("2. Extract the zip file")
            print("3. Create folder at C:\\ffmpeg")
            print("4. Copy contents into C:\\ffmpeg")
            print("5. Add C:\\ffmpeg\\bin to your PATH environment variable")
            return []
            
        # Create a file to track downloaded videos
        downloaded_videos_file = Config.SCORING_DIR / "downloaded_videos.json"
        downloaded_videos = set()
        if downloaded_videos_file.exists():
            try:
                with open(downloaded_videos_file, 'r') as f:
                    downloaded_videos = set(json.load(f))
            except:
                pass

        videos = []
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(Config.SCORING_DIR / 'audio' / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        # Enhanced search queries with band scores
        enhanced_queries = []
        band_scores = ['6.0', '6.5', '7.0', '7.5', '8.0', '8.5']
        parts = ['part 1', 'part 2', 'part 3']
        
        for query in Config.YOUTUBE_QUERIES[:max_videos]:
            for band in band_scores:
                for part in parts:
                    enhanced_queries.append(f"{query} band {band} {part}")
        
        random.shuffle(enhanced_queries)  # Randomize query order
        videos_needed = max_videos * 2  # Get more videos than needed to allow for filtering
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for query in enhanced_queries:
                if len(videos) >= videos_needed:
                    break
                    
                try:
                    # Search more results per query
                    search_results = ydl.extract_info(f"ytsearch10:{query}", download=False)
                    
                    if not search_results or 'entries' not in search_results:
                        continue
                        
                    for entry in search_results['entries']:
                        if len(videos) >= videos_needed:
                            break
                            
                        video_id = entry['id']
                        title = entry['title']
                        duration = entry.get('duration', 0)
                        
                        # Skip if already downloaded
                        if video_id in downloaded_videos:
                            continue
                            
                        # Enhanced video filtering criteria
                        if (
                            any(word in title.lower() for word in ['ielts', 'speaking', 'band']) and
                            duration >= 180 and duration <= 900 and  # 3-15 minutes
                            re.search(r'band\s*\d+(?:\.\d+)?', title.lower()) and  # Must have band score
                            not any(word in title.lower() for word in ['preparation', 'tips', 'tricks'])  # Exclude non-test videos
                        ):
                            print(f"  📥 Downloading: {title[:50]}...")
                            
                            try:
                                # Download video
                                ydl.download([f"https://youtube.com/watch?v={video_id}"])
                                
                                # Get transcript first to verify it's available
                                transcript = self.get_youtube_transcript(video_id)
                                if not transcript:
                                    print(f"    ⚠️ No transcript available, skipping...")
                                    continue
                                
                                videos.append({
                                    'id': video_id,
                                    'title': title,
                                    'url': f"https://youtube.com/watch?v={video_id}",
                                    'duration': duration
                                })
                                
                                # Add to downloaded videos tracker
                                downloaded_videos.add(video_id)
                                
                            except Exception as e:
                                print(f"    ⚠️ Error downloading: {str(e)}")
                                continue
                            
                except Exception as e:
                    print(f"  ⚠️ Error searching {query}: {str(e)}")
                    continue
                    
        # Save updated downloaded videos list
        try:
            with open(downloaded_videos_file, 'w') as f:
                json.dump(list(downloaded_videos), f)
        except Exception as e:
            print(f"  ⚠️ Error saving downloaded videos list: {str(e)}")
                    
        return videos
    
    def get_youtube_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript from YouTube or generate it from audio using Whisper"""
        transcript_path = Config.SCORING_DIR / "transcripts" / f"{video_id}.txt"
        
        # First try to get transcript from YouTube
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([t['text'] for t in transcript_list])
            
            # Save transcript
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
                
            print(f"    ✅ Got transcript from YouTube")
            return full_text
        except Exception as e:
            print(f"    ⚠️ No YouTube transcript available: {str(e)}")
            
        # If YouTube transcript not available, try using Whisper on the audio file
        try:
            audio_path = Config.SCORING_DIR / "audio" / f"{video_id}.wav"
            if audio_path.exists():
                print(f"    🎯 Generating transcript using Whisper...")
                full_text = self.transcribe_audio(str(audio_path))
                
                # Save transcript
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                    
                print(f"    ✅ Generated transcript using Whisper")
                return full_text
            else:
                print(f"    ⚠️ Audio file not found: {audio_path}")
                return None
        except Exception as e:
            print(f"    ⚠️ Error generating transcript with Whisper: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        if self.whisper_model is None:
            print("📝 Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def extract_band_score(self, text: str, title: str = "") -> Dict[str, float]:
        """Extract band scores from text and title using patterns"""
        scores = {
            'band_overall': None,
            'fluency': None,
            'grammar': None,
            'lexical': None,
            'pronunciation': None
        }
        
        # Pattern matching for band scores
        patterns = {
            'band_overall': [
                r'band\s*(?:score\s*)?(?:is\s*)?(\d+(?:\.\d+)?)',
                r'(?:speaking|ielts)\s*band\s*(\d+(?:\.\d+)?)',
                r'band\s*level\s*(\d+(?:\.\d+)?)'
            ],
            'fluency': [
                r'fluency.*?(\d+(?:\.\d+)?)',
                r'fluency\s*and\s*coherence:\s*(\d+(?:\.\d+)?)',
            ],
            'grammar': [
                r'grammar.*?(\d+(?:\.\d+)?)',
                r'grammatical\s*(?:range|accuracy):\s*(\d+(?:\.\d+)?)',
                r'grammar\s*(?:score|mark):\s*(\d+(?:\.\d+)?)'
            ],
            'lexical': [
                r'lexical.*?(\d+(?:\.\d+)?)',
                r'vocabulary.*?(\d+(?:\.\d+)?)',
                r'lexical\s*resource:\s*(\d+(?:\.\d+)?)'
            ],
            'pronunciation': [
                r'pronunciation.*?(\d+(?:\.\d+)?)',
                r'pronounc.*?(\d+(?:\.\d+)?)'
            ]
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                try:
                    scores[key] = float(match.group(1))
                except:
                    pass
                    
        return scores
    
    def generate_scoring_metadata(self, videos: List[Dict]) -> pd.DataFrame:
        """Generate metadata for scoring dataset"""
        print("📊 Generating scoring metadata...")
        
        data = []
        for idx, video in enumerate(tqdm(videos)):
            video_id = video['id']
            
            # Check if audio exists
            audio_path = Config.SCORING_DIR / "audio" / f"{video_id}.wav"
            if not audio_path.exists():
                continue
                
            # Get or generate transcript
            transcript_path = Config.SCORING_DIR / "transcripts" / f"{video_id}.txt"
            if transcript_path.exists():
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            else:
                transcript = self.transcribe_audio(str(audio_path))
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcript)
            
            # Extract scores (simplified - in real scenario, you'd need manual annotation)
            scores = self.extract_band_score(video['title'])
            
            # Split into parts with topics
            parts = self.split_speaking_parts(transcript, video['title'])
            
            for part_data in parts:
                part_num = part_data['part']
                data.append({
                    'id': f"{video_id}_{part_num:04d}",
                    'part': part_num,
                    'topic': part_data['topic'],
                    'question': part_data['question'],
                    'answer': part_data['answer'],
                    'transcript': transcript,  # Keep full transcript for context
                    'band_overall': scores['band_overall'],
                    'fluency': scores['fluency'],
                    'grammar': scores['grammar'],
                    'lexical': scores['lexical'],
                    'pronunciation': scores['pronunciation'],
                    'feedback': self.generate_feedback(scores, part_num),
                    'audio_path': f"audio/{video_id}.wav"
                })
                
        return pd.DataFrame(data)
    
    def split_speaking_parts(self, transcript: str, title: str = "") -> List[Dict[str, str]]:
        """Split transcript into question-answer pairs with topics"""
        parts = []
        combined_text = f"{title}\n{transcript}"
        
        # Topic patterns
        topic_patterns = {
            "Work": r'(?:work|job|career|profession|company|office)',
            "Study": r'(?:study|education|school|university|college|course)',
            "Hometown": r'(?:hometown|city|village|live|neighborhood)',
            "Family": r'(?:family|parents|siblings|relatives|children)',
            "Hobbies": r'(?:hobbies?|free time|interests|activities|enjoy)',
            "Technology": r'(?:technology|computer|internet|phone|digital)',
            "Environment": r'(?:environment|nature|pollution|climate|weather)',
            "Food": r'(?:food|cooking|eat|cuisine|meal|restaurant)',
            "Travel": r'(?:travel|journey|trip|holiday|vacation|abroad)',
            "Sports": r'(?:sports?|exercise|fitness|game|team|play)',
            "Entertainment": r'(?:entertainment|movies?|music|television|shows)',
            "Culture": r'(?:culture|tradition|customs|festival|celebration)'
        }
        
        # Part 1 patterns - simple Q&A
        part1_patterns = [
            r'(?:do you|can you|what|where|when|why|how|describe|tell me about|are you)\s+[^.?!]+[.?!]',
            r'(?:would you|have you|could you|is there|are there)\s+[^.?!]+[.?!]'
        ]
        
        # Part 2 patterns - cue card
        part2_markers = [
            r'cue card',
            r'talk about',
            r'describe',
            r'you should say:?',
            r'you should include:?'
        ]
        
        # Part 3 patterns - discussion
        part3_patterns = [
            r'(?:what do you think|in your opinion|why do you think|how do you feel)\s+[^.?!]+[.?!]',
            r'(?:do you agree|what are the|what is the|how important)\s+[^.?!]+[.?!]'
        ]
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in combined_text.split('\n') if p.strip()]
        
        current_part = 1
        current_topic = "General"
        
        for i, paragraph in enumerate(paragraphs):
            # Detect topic
            for topic, pattern in topic_patterns.items():
                if re.search(pattern, paragraph.lower()):
                    current_topic = topic
                    break
            
            # Part 2 Cue Card detection
            is_cue_card = any(re.search(marker, paragraph.lower()) for marker in part2_markers)
            if is_cue_card:
                current_part = 2
                bullet_points = re.findall(r'[•-]\s*([^•-]+)', paragraph)
                if bullet_points:
                    question = paragraph
                    # Look for the answer in subsequent paragraphs
                    answer = ""
                    j = i + 1
                    while j < len(paragraphs) and not any(re.search(p3p, paragraphs[j].lower()) for p3p in part3_patterns):
                        answer += paragraphs[j] + " "
                        j += 1
                    parts.append({
                        'part': 2,
                        'topic': current_topic,
                        'question': question.strip(),
                        'answer': answer.strip()
                    })
                continue
            
            # Part 1 and 3 Q&A
            question_patterns = part1_patterns if current_part == 1 else part3_patterns
            for pattern in question_patterns:
                questions = re.findall(pattern, paragraph)
                for q in questions:
                    q_end = paragraph.find(q) + len(q)
                    answer = paragraph[q_end:].strip()
                    if not answer and i + 1 < len(paragraphs):
                        answer = paragraphs[i + 1]
                    
                    # Part 3 detection
                    if current_part == 1 and any(re.search(p3p, q.lower()) for p3p in part3_patterns):
                        current_part = 3
                    
                    parts.append({
                        'part': current_part,
                        'topic': current_topic,
                        'question': q.strip(),
                        'answer': answer.strip()
                    })
        
        # Ensure we have at least one part
        if not parts:
            parts.append({
                'part': 1,
                'topic': 'General',
                'question': 'Tell me about yourself.',
                'answer': transcript[:500].strip()
            })
        
        return parts
    
    def generate_feedback(self, scores: Dict[str, float], part: int) -> str:
        """Generate detailed feedback based on scores and speaking part"""
        overall = scores['band_overall']
        feedback = []
        
        # Part-specific feedback
        part_criteria = {
            1: ("short answers", "natural responses", "personal information"),
            2: ("structured description", "relevant points", "coherent narrative"),
            3: ("abstract discussion", "opinion development", "analytical depth")
        }
        
        # Generate overall band feedback
        if overall >= 8.0:
            feedback.append("Excellent performance demonstrating full operational command of the language")
        elif overall >= 7.0:
            feedback.append("Good command of the language with occasional inaccuracies")
        elif overall >= 6.0:
            feedback.append("Generally effective command with some limitations")
        else:
            feedback.append("Basic communication with notable limitations")
        
        # Component-specific feedback
        if scores['fluency'] >= 7.0:
            feedback.append("Speaks fluently with only occasional hesitation")
        elif scores['fluency'] < 6.0:
            feedback.append("Needs improvement in speaking fluency and coherence")
            
        if scores['grammar'] >= 7.0:
            feedback.append("Uses a range of complex structures with good control")
        elif scores['grammar'] < 6.0:
            feedback.append("Should work on grammatical accuracy and range")
            
        if scores['lexical'] >= 7.0:
            feedback.append("Demonstrates good command of vocabulary")
        elif scores['lexical'] < 6.0:
            feedback.append("Needs to expand vocabulary range")
            
        if scores['pronunciation'] >= 7.0:
            feedback.append("Clear pronunciation with good control of features")
        elif scores['pronunciation'] < 6.0:
            feedback.append("Should focus on improving pronunciation clarity")
        
        # Add part-specific feedback
        if part in part_criteria:
            criteria = part_criteria[part]
            if overall >= 7.0:
                feedback.append(f"Strong performance in Part {part} showing good control of {criteria[1]}")
            else:
                feedback.append(f"In Part {part}, focus on improving {criteria[0]} and {criteria[2]}")
        
        return " | ".join(feedback)

# ================== Dataset B: Practice Q&A Dataset ==================

class PracticeDataGenerator:
    def __init__(self):
        self.setup_directories()
        self.setup_api()
        
    def setup_api(self):
        """Setup Gemini API with current key"""
        if Config.GEMINI_API_KEY:
            try:
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"⚠️ Error configuring Gemini API: {str(e)}")
                self.model = None
        else:
            self.model = None
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(Config.PRACTICE_DIR, exist_ok=True)
        os.makedirs(Config.PRACTICE_DIR / "sample_answers", exist_ok=True)
        
    def load_ielts_questions(self) -> List[Dict]:
        """Load IELTS questions from various sources"""
        questions = []
        
        # Part 1 Questions
        part1_questions = [
            {"part": 1, "topic": "Hometown", "question": "Where is your hometown?"},
            {"part": 1, "topic": "Hometown", "question": "Do you like your hometown?"},
            {"part": 1, "topic": "Work/Study", "question": "Do you work or are you a student?"},
            {"part": 1, "topic": "Work/Study", "question": "What do you study?"},
            {"part": 1, "topic": "Technology", "question": "Do you like using technology?"},
            {"part": 1, "topic": "Technology", "question": "What technology do you use every day?"},
            {"part": 1, "topic": "Food", "question": "What's your favorite food?"},
            {"part": 1, "topic": "Food", "question": "Do you like cooking?"},
            {"part": 1, "topic": "Sports", "question": "Do you like sports?"},
            {"part": 1, "topic": "Sports", "question": "What sports are popular in your country?"},
        ]
        
        # Part 2 Questions (Cue Cards)
        part2_questions = [
            {
                "part": 2, 
                "topic": "People",
                "question": """Describe a person who has influenced you.
You should say:
• Who this person is
• How you know them
• What they have done
• And explain why they influenced you

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            },
            {
                "part": 2,
                "topic": "Places",
                "question": """Describe a place you would like to visit.
You should say:
• Where it is
• What you know about it
• What you would do there
• And explain why you want to visit this place

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            },
            {
                "part": 2,
                "topic": "Events",
                "question": """Describe a memorable event in your life.
You should say:
• When it happened
• Where it happened
• Who was there
• And explain why it was memorable

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            },
            {
                "part": 2,
                "topic": "Objects",
                "question": """Describe something you own that is important to you.
You should say:
• What it is
• How you got it
• How often you use it
• And explain why it is important to you

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            },
            {
                "part": 2,
                "topic": "Technology",
                "question": """Describe a piece of technology you use regularly.
You should say:
• What it is
• How long you have been using it
• What you use it for
• And explain why it is important to you

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            },
            {
                "part": 2,
                "topic": "Books",
                "question": """Describe a book that has made a strong impression on you.
You should say:
• What kind of book it is
• When you read it
• What it is about
• And explain why it made such an impression on you

You will have to speak about the topic for 1-2 minutes.
You have 1 minute to think about what you're going to say."""
            }
        ]
        
        # Part 3 Questions
        part3_questions = [
            {"part": 3, "topic": "Education", "question": "How has education changed in your country in the last 20 years?"},
            {"part": 3, "topic": "Technology", "question": "What are the advantages and disadvantages of technology in education?"},
            {"part": 3, "topic": "Environment", "question": "What can individuals do to protect the environment?"},
            {"part": 3, "topic": "Culture", "question": "How important is it to preserve traditional culture?"},
        ]
        
        questions.extend(part1_questions)
        questions.extend(part2_questions)
        questions.extend(part3_questions)
        
        return questions
    
    def generate_sample_answer_claude(self, question: str, part: int, band: float) -> str:
        """Generate sample answer using Gemini API"""
        if not self.model:
            print("⚠️ No active model. Trying to set up API...")
            self.setup_api()
            if not self.model:
                print("⚠️ Could not set up API. Using template generation.")
                return self.generate_sample_answer_template(question, part, band)
        
        max_retries = 3  # Limit retries to avoid excessive API calls
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # For Part 2, ensure we handle bullet points properly
                if part == 2:
                    bullet_points = re.findall(r'•\s*([^•\n]+)', question)
                    if bullet_points:
                        bullet_points_str = "\n".join(f"- {point.strip()}" for point in bullet_points)
                        question = re.sub(r'(?s)You should say:.*?(?=\n\n|$)', f'You should address:\n{bullet_points_str}', question)
                
                prompt = f"""Generate a natural IELTS Speaking response for Band {band}.

Context:
- Part: {part} {'(Cue Card - 2 minute long answer)' if part == 2 else '(Short answer)' if part == 1 else '(Discussion)'}
- Question: {question}
- Target Band: {band}

Response Requirements:
1. Fluency: {'Very fluent with rare hesitation' if band >= 7.5 else 'Generally fluent with some pauses' if band >= 6.5 else 'Basic fluency with noticeable hesitation'}
2. Vocabulary: {'Sophisticated and precise' if band >= 7.5 else 'Good range with some natural collocations' if band >= 6.5 else 'Simple but adequate'}
3. Grammar: {'Complex structures accurately used' if band >= 7.5 else 'Mix of complex and simple structures' if band >= 6.5 else 'Basic structures with some errors'}
4. Length: {'200-250 words with clear structure' if part == 2 else '50-70 words direct response' if part == 1 else '100-150 words developed answer'}
5. Style: {'Natural flow with good topic development' if band >= 7 else 'Clear message with some development' if band >= 6 else 'Basic communication'}

Natural Speaking Features to Include:
{'''- Sophisticated linking (moreover, furthermore)
- Idiomatic expressions
- Complex comparisons
- Varied sentence structures''' if band >= 7.5 else '''- Basic linking words (also, however)
- Some hesitation markers (well, you know)
- Simple examples
- Occasional self-correction''' if band >= 6.5 else '''- Simple connectors (and, but, so)
- Hesitation fillers (um, uh)
- Basic examples
- Simple opinions'''}

Response:"""

                response = self.model.generate_content(prompt)
                return response.text.strip()
                
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ API error (attempt {retries + 1}/{max_retries}): {last_error}")
                
                # Check for quota exceeded
                if "quota exceeded" in last_error.lower() or "429" in last_error:
                    print("⚠️ API quota exceeded. Using template generation.")
                    break
                
                retries += 1
                if retries < max_retries:
                    print("Trying again after short delay...")
                    time.sleep(2)  # Add small delay between retries
                    self.setup_api()  # Try to refresh the API setup
        
        print(f"⚠️ API generation failed after {retries} attempts. Using template generation.")
        return self.generate_sample_answer_template(question, part, band)
    
    def generate_sample_answer_template(self, question: str, part: int, band: float) -> str:
        """Generate template-based answer when API is not available"""
        # Extract topic
        topic = ""
        if "describe" in question.lower():
            topic_match = re.search(r'describe\s+(?:a|an)\s+([^.]+)', question.lower())
            if topic_match:
                topic = topic_match.group(1).strip()
        
        if not topic:
            # Try to extract topic from question
            words = [w for w in question.lower().split() 
                    if len(w) > 3 and w not in {
                        'what', 'when', 'where', 'which', 'how', 'why',
                        'tell', 'about', 'describe', 'would', 'could', 'should'
                    }]
            topic = words[0] if words else "this topic"
        
        # Extract bullet points for Part 2
        bullet_points = re.findall(r'•\s*([^•\n]+)', question)
        
        # Part 1 - Short direct answers
        if part == 1:
            if band >= 7.0:
                return (
                    f"That's an interesting question. From my perspective, {topic} plays "
                    f"a significant role in my personal experience. Specifically, I find "
                    f"it quite beneficial because it helps me in various aspects of my "
                    f"daily life. For instance, I regularly engage with this in my "
                    f"routine activities."
                )
            elif band >= 6.0:
                return (
                    f"Well, I would say that {topic} is quite important to me. I usually "
                    f"spend time on this, and I find it helpful for different reasons. "
                    f"For example, it helps me with my daily activities."
                )
            else:
                return (
                    f"Um... I think {topic} is okay. Sometimes I... you know... use it, "
                    f"and yes, it's useful for me. I like it because... um... it helps "
                    f"me do things."
                )
        
        # Part 2 - Cue card long answers
        elif part == 2:
            # Handle cue card responses
            if bullet_points:
                if band >= 7.0:
                    answer_parts = [
                        f"I'd like to tell you about {topic}, which is something that "
                        f"holds particular significance for me."
                    ]
                    for point in bullet_points:
                        answer_parts.append(
                            f"\nRegarding {point.strip()}, I can say that this aspect "
                            f"is particularly noteworthy and has made a strong impression "
                            f"on me."
                        )
                    answer_parts.append(
                        f"\nIn conclusion, this experience has had a profound impact on "
                        f"my perspective and continues to influence me today."
                    )
                elif band >= 6.0:
                    answer_parts = [f"I want to talk about {topic}."]
                    for point in bullet_points:
                        answer_parts.append(
                            f"\nWhen it comes to {point.strip()}, I would say that "
                            f"it's quite important and meaningful to me."
                        )
                    answer_parts.append(
                        f"\nThat's why this {topic} matters so much in my life."
                    )
                else:
                    answer_parts = [f"Um... I will talk about {topic}."]
                    for point in bullet_points:
                        answer_parts.append(
                            f"\nAbout {point.strip()}... well... I think... it's good "
                            f"because... um... it helps me."
                        )
                    answer_parts.append("\nSo... that's my answer about this topic.")
                return "".join(answer_parts)
            else:
                return (
                    f"I would like to describe {topic}. This is something that I find "
                    f"particularly interesting and meaningful. Based on my personal "
                    f"experience, I have several points to share about this topic. "
                    f"First, let me tell you about my main impressions..."
                )
        
        # Part 3 - Discussion answers
        else:
            if band >= 7.0:
                return (
                    f"This is indeed a complex matter that deserves careful consideration. "
                    f"From my perspective, {topic} has several important implications "
                    f"for society. Firstly, we must consider its immediate impact on "
                    f"individuals. Moreover, looking at the broader picture, I believe "
                    f"this issue will become increasingly significant in the future."
                )
            elif band >= 6.0:
                return (
                    f"I think {topic} is an important issue to discuss. There are both "
                    f"advantages and disadvantages to consider. The main benefit is its "
                    f"positive impact on people's lives, although there can be some "
                    f"challenges that need to be addressed."
                )
            else:
                return (
                    f"I think {topic} is... um... important. There are good things "
                    f"and... well... some bad things too. Some people like it, but "
                    f"others don't. In my opinion, it's mostly good for everyone."
                )
        # Part 2 - Cue card response
        if part == 2:
            if bullet_points:
                # Generate structured response based on bullet points
                answer_parts = []
                if band >= 7.0:
                    answer_parts.append(f"I'd like to tell you about {topic}, which is something that holds particular significance for me.")
                    for point in bullet_points:
                        point = point.strip()
                        answer_parts.append(f"\nRegarding {point}, I can say that this aspect is particularly noteworthy. ")
                    answer_parts.append("\nOverall, this experience has had a lasting impact on my perspective.")
                elif band >= 6.0:
                    answer_parts.append(f"I want to talk about {topic}.")
                    for point in bullet_points:
                        point = point.strip()
                        answer_parts.append(f"\nAbout {point}, I would say that it's important. ")
                    answer_parts.append("\nThat's why this matters to me.")
                else:
                    answer_parts.append(f"Um... I will talk about {topic}.")
                    for point in bullet_points:
                        point = point.strip()
                        answer_parts.append(f"\nFor {point}... well... I think... ")
                    answer_parts.append("\nSo... that's my answer.")
                return "".join(answer_parts)
            else:
                return f"I want to describe {topic}. It's something that I find interesting. I have some experience with it, and I think it's important for several reasons."
        
        # Part 3 - Discussion
        else:
            if band >= 7.0:
                return f"This is indeed a complex matter that deserves careful consideration. From my perspective, {topic} has several important implications for society. Firstly, we must consider its immediate impact on individuals. Moreover, looking at the broader picture, I believe this issue will become increasingly significant in the future."
            elif band >= 6.0:
                return f"I think {topic} is an important issue to discuss. There are both advantages and disadvantages to consider. The main benefit is its positive impact, although there can be some challenges too."
            else:
                return f"I think {topic} is... um... important. There are good things and... well... some bad things too. Some people like it, but others don't. In my opinion, it's mostly good."

        # Apply replacements if template is used
        if 'template' in locals():
            replacements = {
                'reasons': 'various factors',
                'future_prediction': 'changes will continue',
                'aspect1': 'certain aspects',
                'contrasting_view': 'there are limitations',
                'personal_connection': 'how it relates to me',
                'context': 'the situation was unique',
                'factor1': 'multiple elements',
                'factor2': 'different circumstances',
                'unexpected_element': 'something surprising happened',
                'impact': 'my understanding',
                'assumption': 'that things are simple',
                'example': 'a specific case',
                'principle': 'the underlying concept',
                'immediate_context': 'the direct impact',
                'broader_impact': 'wider implications',
                'nuanced_conclusion': 'a balanced approach is needed'
            }
            
            for key, value in replacements.items():
                template = template.replace(f'{{{key}}}', value)
            return template
            
        return ""  # Return empty string if no template was used
    
    def extract_keywords(self, text: str, topic: str) -> str:
        """Extract keywords from sample answer"""
        # Simple keyword extraction (in production, use NLP library)
        words = text.lower().split()
        
        # Filter common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'can', 'that', 'this', 'these', 'those',
                       'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her'}
        
        keywords = []
        word_count = {}
        
        for word in words:
            word = word.strip('.,!?";')
            if word and word not in common_words and len(word) > 3:
                word_count[word] = word_count.get(word, 0) + 1
                
        # Get top 10 most frequent words
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:10]]
        
        return ', '.join(keywords)
    
    def generate_practice_metadata(self) -> pd.DataFrame:
        """Generate complete practice dataset"""
        print("📝 Generating practice Q&A dataset...")
        
        questions = self.load_ielts_questions()
        bands = [2.0, 3.0, 3.5,4.0,4.5, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
        
        data = []
        total = len(questions) * len(bands)
        
        with tqdm(total=total, desc="Generating answers") as pbar:
            for q_idx, question_data in enumerate(questions):
                for band in bands:
                    # Generate sample answer
                    sample_answer = self.generate_sample_answer_claude(
                        question_data['question'],
                        question_data['part'],
                        band
                    )
                    
                    # Extract keywords
                    keywords = self.extract_keywords(sample_answer, question_data['topic'])
                    
                    # Create data entry
                    data.append({
                        'id': f"P{q_idx:04d}_B{int(band*10):03d}",
                        'part': question_data['part'],
                        'question': question_data['question'],
                        'band': band,
                        'sample_answer': sample_answer,
                        'topic': question_data['topic'],
                        'keywords': keywords
                    })
                    
                    # Save sample answer to file
                    answer_path = Config.PRACTICE_DIR / "sample_answers" / f"P{q_idx:04d}_B{int(band*10):03d}.txt"
                    with open(answer_path, 'w', encoding='utf-8') as f:
                        f.write(f"Question: {question_data['question']}\n")
                        f.write(f"Band: {band}\n")
                        f.write(f"Topic: {question_data['topic']}\n\n")
                        f.write(f"Sample Answer:\n{sample_answer}\n\n")
                        f.write(f"Keywords: {keywords}\n")
                    
                    pbar.update(1)
                    time.sleep(0.1)  # Rate limiting
                    
        return pd.DataFrame(data)

# ================== Main Pipeline ==================

class IELTSDatasetBuilder:
    def __init__(self):
        self.scoring_collector = ScoringDataCollector()
        self.practice_generator = PracticeDataGenerator()
        
    def check_missing_transcripts(self):
        """Check for audio files without transcripts and generate them"""
        print("🔍 Checking for missing transcripts...")
        
        # Get all audio files
        audio_files = list(Config.SCORING_DIR.glob(f"audio/*.{Config.AUDIO_FORMAT}"))
        total_audio = len(audio_files)
        
        if total_audio == 0:
            print("No audio files found.")
            return
            
        missing_transcripts = []
        for audio_file in audio_files:
            video_id = audio_file.stem
            transcript_path = Config.SCORING_DIR / "transcripts" / f"{video_id}.txt"
            
            if not transcript_path.exists():
                missing_transcripts.append(video_id)
                
        print(f"\nFound {len(missing_transcripts)} audio files without transcripts out of {total_audio} total files.")
        
        if missing_transcripts:
            print("\n📝 Generating missing transcripts...")
            for video_id in tqdm(missing_transcripts, desc="Generating transcripts"):
                self.scoring_collector.get_youtube_transcript(video_id)
                
        return len(missing_transcripts)
    
    def build_complete_dataset(self):
        """Build complete IELTS Speaking dataset"""
        print("🚀 Starting IELTS Speaking Dataset Builder")
        print("=" * 50)
        
        # Get user input for configuration
        download_videos = input("Download new videos? (y/n): ").lower() == 'y'
        if download_videos:
            max_videos = int(input("How many videos to download? (1-20): "))
            max_videos = max(1, min(20, max_videos))  # Limit between 1-20
        else:
            max_videos = 0
            
        generate_practice = input("Generate practice dataset? (y/n): ").lower() == 'y'
        
        # Part A: Scoring Dataset
        scoring_df = None
        if download_videos:
            print("\n📊 Building Scoring Dataset (A)")
            print("-" * 30)
            
            try:
                # Download videos
                videos = self.scoring_collector.download_youtube_videos(max_videos)
                if videos:
                    print(f"✅ Downloaded {len(videos)} videos")
                    
                    # Generate metadata
                    scoring_df = self.scoring_collector.generate_scoring_metadata(videos)
                    
                    if not scoring_df.empty:
                        # Save metadata
                        scoring_csv_path = Config.SCORING_DIR / "metadata.csv"
                        scoring_df.to_csv(scoring_csv_path, index=False, encoding='utf-8')
                        print(f"✅ Saved scoring metadata: {len(scoring_df)} entries")
                    else:
                        print("⚠️ No scoring data was generated")
                else:
                    print("⚠️ No videos were downloaded. Skipping metadata generation.")
            except Exception as e:
                print(f"⚠️ Error in scoring dataset generation: {str(e)}")
                scoring_df = pd.DataFrame()
            
            # Display sample
            print("\n📋 Sample scoring data:")
            if scoring_df is not None and not scoring_df.empty:
                available_columns = scoring_df.columns.tolist()
                display_columns = [col for col in ['id', 'part', 'band_overall', 'fluency', 'grammar'] if col in available_columns]
                if display_columns:
                    print(scoring_df[display_columns].head())
                else:
                    print("No data available to display")
            else:
                print("No scoring data was collected")
        
        # Part B: Practice Dataset
        if generate_practice:
            print("\n📚 Building Practice Q&A Dataset (B)")
            print("-" * 30)
            
            # Generate practice data
            practice_df = self.practice_generator.generate_practice_metadata()
            
            # Save metadata
            practice_csv_path = Config.PRACTICE_DIR / "metadata.csv"
            practice_df.to_csv(practice_csv_path, index=False, encoding='utf-8')
            print(f"✅ Saved practice metadata: {len(practice_df)} entries")
            
            # Display sample
            print("\n📋 Sample practice data:")
            print(practice_df[['id', 'part', 'band', 'topic']].head())
        
        # Summary statistics
        self.print_summary_statistics(scoring_df if download_videos else None, 
                                     practice_df if generate_practice else None)
        
    def print_summary_statistics(self, 
                                scoring_df: Optional[pd.DataFrame], 
                                practice_df: Optional[pd.DataFrame]):
        """Print summary statistics"""
        print("\n" + "=" * 50)
        print("📊 DATASET SUMMARY")
        print("=" * 50)
        
        if scoring_df is not None and not scoring_df.empty:
            print("\n🎯 Scoring Dataset:")
            print(f"  - Total entries: {len(scoring_df)}")
            if 'part' in scoring_df.columns:
                print(f"  - Parts covered: {scoring_df['part'].unique()}")
            if 'band_overall' in scoring_df.columns:
                print(f"  - Band range: {scoring_df['band_overall'].min():.1f} - {scoring_df['band_overall'].max():.1f}")
                print(f"  - Average band: {scoring_df['band_overall'].mean():.2f}")
            else:
                print("  - No scoring data available")
        if practice_df is not None:
            print("\n📚 Practice Dataset:")
            print(f"  - Total entries: {len(practice_df)}")
            print(f"  - Questions: {practice_df['question'].nunique()}")
            print(f"  - Topics: {practice_df['topic'].nunique()}")
            print(f"  - Band levels: {sorted(practice_df['band'].unique())}")
            
            # Topic distribution
            topic_dist = practice_df.groupby('topic').size()
            print(f"\n  Topic Distribution:")
            for topic, count in topic_dist.items():
                print(f"    - {topic}: {count} samples")

# ================== Utility Functions ==================

def setup_environment():
    """Setup environment and install required packages"""
    print("🔧 Setting up environment...")
    
    requirements = [
        "yt-dlp",
        "openai-whisper",
        "anthropic",
        "pydub",
        "pandas",
        "tqdm",
        "youtube-transcript-api"
    ]
    
    print("Required packages:")
    for req in requirements:
        print(f"  - {req}")
    
    print("\nInstall with: pip install " + " ".join(requirements))
    
def validate_api_keys():
    """Validate API keys are set"""
    if not Config.GEMINI_API_KEY:
        print("⚠️ Warning: GEMINI_API_KEY not set. Will use template-based generation.")
        print("   Set with: export GEMINI_API_KEY='your-key-here'")
        return False
    return True

# ================== Main Execution ==================

def main():
    """Main execution function"""
    try:
        # Setup
        setup_environment()
        validate_api_keys()
        
        # Build dataset
        builder = IELTSDatasetBuilder()
        
        # Check for missing transcripts first
        missing_count = builder.check_missing_transcripts()
        if missing_count > 0:
            print(f"\n✅ Generated {missing_count} missing transcripts")
        
        # Build dataset with user configuration
        builder.build_complete_dataset()
        
        print("\n✅ Dataset building complete!")
        print(f"📁 Output directory: {Config.BASE_DIR}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
    finally:
        # Save any pending changes
        print("\n🔄 Cleaning up...")

if __name__ == "__main__":
    main()
