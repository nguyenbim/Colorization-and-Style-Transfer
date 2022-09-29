from typing import Union,Dict
from pydantic import BaseModel
import pydantic

class Polygon_Plate(BaseModel):
    vehicle_type: str = None
    id : str = None  
    recognized_data : Dict[str, Union[list,float, str]]


class ImagesFromCLients(BaseModel):
    img_data_str : str = None
    class Config: 
        schema_extra = {
                'example': {
                    'image': ('iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAIAAAACUFjqAAAACXBIWXMAAC4jAAAuIwF4pT92AAAA'
                                'B3RJTUUH5AwZAyMzqt+uDgAAABVJREFUGNNj/P//PwNuwMSAF4xUaQCl4wMR/9A5uQAAAABJRU5E'
                                'rkJggg==')
                }
            }