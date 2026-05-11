import os
from services.youtube_service import YouTubeExtractorService

os.makedirs('documents/test_ns', exist_ok=True)
urls = ['https://www.youtube.com/watch?v=jNQXAC9IVRw', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'] # 'Me at the zoo' and 'Rick Roll'
for url in urls:
    print(f"Extracting {url}...")
    text = YouTubeExtractorService.get_transcript(url)
    if text:
        vid = YouTubeExtractorService.extract_video_id(url)
        file_path = YouTubeExtractorService.save_as_pdf(text, vid, 'documents/test_ns')
        print(f"Saved: {file_path}")
    else:
        print(f"Failed to extract text for {url}")
