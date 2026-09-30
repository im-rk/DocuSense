from typing import Optional, Union

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    name: str = ""
    qty: Optional[float] = None
    price: Optional[float] = None


class FieldsUpdate(BaseModel):
    """Human corrections. Only the listed keys are changed; a null/empty value removes the field."""
    fields: dict[str, Optional[Union[str, float]]] = Field(default_factory=dict)
    line_items: Optional[list[LineItem]] = None


class UploadResponse(BaseModel):
    document_id: str
    status: str
    message: str = "Processing started"
