import asyncio
import os
from youtube_transcript_api import YouTubeTranscriptApi
from agents import Agent, Runner, function_tool, ItemHelpers, OpenAIChatCompletionsModel, trace, AsyncOpenAI
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from agents.run import RunConfig


load_dotenv()
API_KEY = os.getenv("API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


@function_tool
def generate_content(video_transcript: str, social_media_platform: str):
    print(f"Generating social media content for: {social_media_platform}...")

    client = AsyncOpenAI(api_key=API_KEY)

    # Generate content

    response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": f"Here is a new video transcript:\n{video_transcript}\n\n"
                                            f"Generate a social media post on my {social_media_platform} based on my provided video transcript.\n"}
            ]  # Increase tokens for longer blog posts
        )
    return response.choices[0].message.content

class Post(BaseModel):
    platform: str
    content: str

content_writer_agent = Agent(
    name="Content Writer Agent",
    instructions="""You are a talented content writer who writes engaging, humorous, informative and 
                    highly readable social media posts. 
                    You will be given a video transcript and social media platforms. 
                    You will generate a social media post based on the video transcript 
                    and the social media platforms.
                    You may search the web for up-to-date information on the topic and 
                    fill in some useful details if needed.""",
    model=model,
    tools=[generate_content,
           ],
    output_type=List[Post]
)

def get_transcript(video_id: str, languages: list = None) -> str:
    """
    Retrieves the transcript for a YouTube video.

    Args:
        video_id (str): The YouTube video ID.
        languages (list, optional): List of language codes to try, in order of preference.
                                   Defaults to ["en"] if None.

    Returns:
        str: The concatenated transcript text.

    Raises:
        Exception: If transcript retrieval fails, with details about the failure.
    """

    if languages is None:
        languages = ["en"]

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=languages)

        transcript_text = " ".join(snippet.text for snippet in fetched_transcript)

        return transcript_text
    
    except Exception as e:
        # Handle specific YouTube transcript API exceptions
        from youtube_transcript_api._errors import (
            CouldNotRetrieveTranscript, 
            VideoUnavailable,
            InvalidVideoId, 
            NoTranscriptFound,
            TranscriptsDisabled
        )

        if isinstance(e, NoTranscriptFound):
            error_msg = f"No transcript found for video {video_id} in languages: {languages}"
        elif isinstance(e, VideoUnavailable):
            error_msg = f"Video {video_id} is unavailable"
        elif isinstance(e, InvalidVideoId):
            error_msg = f"Invalid video ID: {video_id}"
        elif isinstance(e, TranscriptsDisabled):
            error_msg = f"Transcripts are disabled for video {video_id}"
        elif isinstance(e, CouldNotRetrieveTranscript):
            error_msg = f"Could not retrieve transcript: {str(e)}"
        else:
            error_msg = f"An unexpected error occurred: {str(e)}"

        print(f"Error: {error_msg}")
        raise Exception(error_msg) from e
    
async def main():
    video_id = "OZ5OZZZ2cvk"
    transcript = get_transcript(video_id)

    msg = f"Generate a LinkedIn post and an Instagram caption based on this video transcript: {transcript}"
    input_items = [{"content": msg, "role": "user"}]

    with trace("Writing content"):
        result = await Runner.run(content_writer_agent, input_items, run_config=config)
        output = ItemHelpers.text_message_outputs(result.new_items)
        print("Generated Post:\n", output)

if __name__ == "__main__":
    asyncio.run(main())