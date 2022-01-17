from typing import List, Optional
from pydantic import BaseModel


class Input_Data_For_Generation(BaseModel):
    text_body: str