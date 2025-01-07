import unittest
from tongagent.tools.text_inspector import TextInspectorTool
class TestTool(unittest.TestCase):
    def test_read_mp3(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3")
        print(result)

    def test_read_excel(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx")
        print(result)
        
    def test_read_excel2(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/5cfb274c-0207-4aa7-9575-6ac0bd95d9b2.xlsx")
        print(result)
        
    def test_read_pptx(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx")
        print(result)
    
    def test_read_docx(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb.docx")
        print(result)
    
    def test_read_pdf(self):
        tool = TextInspectorTool()
        result = tool.forward(file_path="data/GAIA/2023/validation/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf")
        print(result)
        # from marker.convert import convert_single_pdf
        # from marker.models import load_all_models
        # model_list = load_all_models()
        # print(model_list)
        # markdown, images, meta_data = convert_single_pdf(
        #     "data/GAIA/2023/validation/67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",
        #     model_list,
        #     max_pages=10)
        # print(markdown)
        
    # def test_read_pdb(self):
    #     tool = TextInspectorTool()
    #     result = tool.forward(file_path="7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb")
    #     print(result)
    
    def test_read_youtube(self):
        tool = TextInspectorTool()
        youtube = "https://www.youtube.com/watch?v=1htKBjuUWec"
        result = tool.forward(file_path=youtube)
        print(result) 
        #best = video.getbest()
        #best.download(quiet=False)
        from vidgear.gears import CamGear
        import cv2
        from moviepy.editor import VideoFileClip
        video_id = youtube[youtube.find("v=")+2:]
        stream = CamGear(
            source=youtube,
            stream_mode=True,
            logging=True,
        ).start()
        
        frame = stream.read()
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(f'{video_id}.mp4', fourcc, stream.framerate, (width, height))
        audio_file = f"{video_id}.mp3"
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
        # video_clip = VideoFileClip(f"{video_id}.mp4")
        # audio_clip = video_clip.audio
        # audio_clip.write_audiofile(audio_file)
        # video_clip.close()
        # audio_clip.close()
        
    
    def test_download(self):
        from tongagent.tools.web_surfer import DownloadTool
        tool = DownloadTool()
        img_addr = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Googleplex_HQ_%28cropped%29.jpg/500px-Googleplex_HQ_%28cropped%29.jpg"
        print(tool.forward(img_addr))
    
    def test_visit(self):
        from tongagent.tools.web_surfer import VisitTool
        tool = VisitTool()
        arxv = "https://arxiv.org/pdf/2407.11522"
        result = tool.forward(arxv)
        print(result)
    
    def test_upload_video(self):
        src = "1htKBjuUWec.mp4"
        target = src
        upload_blob(
            "agent-tuning",
            src,
            target
        )
    
    def test_video(self):
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        # TODO(developer): Update project_id and location
        PROJECT_ID = "agenttuning"
        vertexai.init(project=PROJECT_ID, location="us-central1")

        model = GenerativeModel("gemini-1.5-flash-001")


        prompt = """Provide a description of the video.
The description should also contain anything important which people say in the video."""
        
        video_file = Part.from_uri(
            uri="gs://agent-tuning/1htKBjuUWec.mp4",
            mime_type="video/mp4",
        )

        contents = [video_file, prompt]

        response = model.generate_content(contents)
        print(response.text)

from google.cloud import storage
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


if __name__ == "__main__":
    unittest.main()