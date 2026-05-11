import logging
import os
import re
import tempfile
import subprocess
import webvtt
from typing import Optional
from fpdf import FPDF

logger = logging.getLogger(__name__)

class YouTubeExtractorService:
    """Service to extract YouTube transcripts and save them as PDFs."""

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract the YouTube video ID from various URL formats."""
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def get_transcript(url: str) -> Optional[str]:
        """Fetch the transcript for a given video URL using yt-dlp."""
        video_id = YouTubeExtractorService.extract_video_id(url)
        if not video_id:
            return None
            
        with tempfile.TemporaryDirectory() as temp_dir:
            # yt-dlp command to download auto or manual subs as vtt
            # we use --write-sub and --write-auto-sub and --sub-langs en
            output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
            
            import sys
            cmd = [
                sys.executable,
                "-m", "yt_dlp",
                "--skip-download",
                "--write-sub",
                "--write-auto-sub",
                "--sub-langs", "en",
                "--sub-format", "vtt",
                "-o", output_template,
                url
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Check for the downloaded .vtt file
                vtt_file = None
                for file in os.listdir(temp_dir):
                    if file.endswith(".vtt"):
                        vtt_file = os.path.join(temp_dir, file)
                        break
                        
                if not vtt_file:
                    logger.error("No VTT subtitle file found for %s", url)
                    return None
                    
                # Parse VTT
                vtt = webvtt.read(vtt_file)
                
                # Extract and deduplicate text lines (auto-subs duplicate lines)
                seen_lines = []
                for caption in vtt:
                    # Clean tags like <00:00:19.039><c>
                    text = re.sub(r'<[^>]+>', '', caption.text)
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (not seen_lines or seen_lines[-1] != line):
                            seen_lines.append(line)
                            
                return " ".join(seen_lines)
                
            except subprocess.CalledProcessError as e:
                logger.error("yt-dlp failed to download subtitles for %s: %s", url, e.stderr)
                return None
            except Exception as e:
                logger.error("Error processing transcript for %s: %s", url, e)
                return None

    @staticmethod
    def save_as_pdf(text: str, video_id: str) -> str:
        """Convert the transcript text to a PDF and save it to a temporary location."""
        # Create a basic PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"YouTube Transcript: {video_id}", ln=True, align='C')
        pdf.ln(10)

        # Body
        pdf.set_font("Arial", size=12)
        
        # Write text, handling long lines with multi_cell
        # fpdf expects latin-1 natively or we should replace unprintable characters
        cleaned_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=cleaned_text)

        # Save
        temp_dir = tempfile.gettempdir()
        filename = f"youtube_{video_id}.pdf"
        file_path = os.path.join(temp_dir, filename)
        pdf.output(file_path)

        return file_path
