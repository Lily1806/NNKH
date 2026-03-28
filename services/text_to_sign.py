import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import Config

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    print("moviepy not installed. Run 'pip install moviepy'")

def text_to_sign_video(text, output_path="output_sign.mp4"):
    """
    Parses text into words, maps to sign language videos in dataraw/train,
    concatenates them using moviepy, and saves to output_path.
    """
    words = text.strip().lower().split()
    clips = []
    
    if not os.path.exists(Config.DATA_RAW_TRAIN):
        print("Raw data directory for videos not found!")
        return None
        
    # Get available classes (handling case sensitivity cautiously)
    available_classes = {d.lower(): d for d in os.listdir(Config.DATA_RAW_TRAIN) 
                         if os.path.isdir(os.path.join(Config.DATA_RAW_TRAIN, d))}
    
    for word in words:
        if word in available_classes:
            actual_dir_name = available_classes[word]
            cls_dir = os.path.join(Config.DATA_RAW_TRAIN, actual_dir_name)
                
            videos = glob.glob(os.path.join(cls_dir, "*.mp4"))
            if videos:
                # Use the first available video for the word
                try:
                    clip = VideoFileClip(videos[0])
                    clips.append(clip)
                except Exception as e:
                    print(f"Error loading clip {videos[0]}: {e}")
            else:
                print(f"No mp4 videos found for word: {word} in {cls_dir}")
        else:
            print(f"No sign language video mapped for word: {word}")
            
    if not clips:
        return None
        
    try:
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(output_path, codec="libx264", audio=False, verbose=False, logger=None)
        
        # Close to free up memory
        for clip in clips:
            clip.close()
        final_video.close()
            
        return output_path
    except Exception as e:
        print(f"Error concatenating clips: {e}")
        return None
