from transformers.agents import Tool
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import os
from vidgear.gears import CamGear
import cv2
from tongagent.utils import CACHE_FOLDER
class VideoQATool(Tool):
    name = "video_visualizer"
    description = "A tool that can answer questions about an attached video."
    inputs = {
        "question": {"description": "the question to answer", "type": "text"},
        "video_path": {
            "description": "The path to the image on which to answer the question",
            "type": "text",
        },
    }
    output_type = "text"
    
    buck_name = "agent-tuning"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PROJECT_ID = "agenttuning"
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-1.5-flash-001")
        self.model = model
    
    def download_youtube_video(self, video_path: str):
        video_id = video_path[video_path.find("v=")+2:]
        stream = CamGear(
            source=video_path,
            stream_mode=True,
            logging=True,
        ).start()
        
        frame = stream.read()
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        local_video_path = f'{CACHE_FOLDER}/{video_id}.mp4'
        video = cv2.VideoWriter(local_video_path, fourcc, stream.framerate, (width, height))
        video.write(frame)
        while frame is not None:
            print("read!")
            frame = stream.read()
            if frame is None:
                break
            video.write(frame)
            # cv2.imshow("Output Frame", frame)
            # # Show output window

            # key = cv2.waitKey(1) & 0xFF
            # if key == ord("q"):
            #     #if 'q' key-pressed break out
            #     break
        cv2.destroyAllWindows()
        # close output window
        video.release()
        # safely close video stream.
        stream.stop()
        return local_video_path
    
    def forward(self, question: str, video_path: str) -> str:
        local_video_path = None
        if video_path.startswith("http") and "youtube" in video_path:
            local_video_path = self.download_youtube_video(video_path)
        
        if os.path.exists(video_path):
            local_video_path = video_path
        
        if local_video_path is None:
            return "This tool does not support this type of video path. Never use this tool to process video again."
        
        fname = local_video_path.split("/")[-1]
        upload_blob(
            self.buck_name,
            local_video_path,
            fname
        )
        gs_path = f"gs://{self.buck_name}/{fname}"
        video_file = Part.from_uri(
            uri=gs_path,
            mime_type="video/mp4",
        )
        contents = [video_file, question]

        response = self.model.generate_content(contents)
        print(response.text)
        return response.text
        

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )