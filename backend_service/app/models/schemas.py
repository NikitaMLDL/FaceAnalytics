from pydantic import BaseModel
from typing import Optional


class FaceRecognizeRequest(BaseModel):
    """
    Model for validating the input data in a face recognition request.

    Attributes:
    - image: str
        A base64-encoded string representing the image, or a path to the image
        that will be used for face recognition. This image is sent to the server
        for processing and recognition.
    """
    image: str  # Base64-encoded string of the image or a file path


class PersonResponse(BaseModel):
    """
    Model for representing data about a recognized person.

    Attributes:
    - name: str
        The name of the recognized person.
    - description: Optional[str]
        A description of the person (e.g., profession or biography).
        Can be empty if no description is available.
    - confidence: float
        The model's confidence in the face recognition, with a value between 0 and 1.
        The higher the value, the more confident the model is that the face belongs to the identified person.
    """
    name: str
    description: Optional[str] = None
    confidence: float
