from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_METADATA_VALUE = "not provided"


class OCRMetadata(BaseModel):
    class_: str = Field(default=DEFAULT_METADATA_VALUE, alias="class")
    level: str = DEFAULT_METADATA_VALUE
    title: str = DEFAULT_METADATA_VALUE
    topic: str = DEFAULT_METADATA_VALUE
    writing_type: str = DEFAULT_METADATA_VALUE

    model_config = {"populate_by_name": True}

    @field_validator("class_", "level", "title", "topic", "writing_type", mode="before")
    @classmethod
    def default_missing_metadata(cls, value):
        if value is None or value == "":
            return DEFAULT_METADATA_VALUE
        return value


class OCRLine(BaseModel):
    line_no: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class OCRSource(BaseModel):
    type: Literal["image", "json"]
    filename: str
    ocr_engine: Literal["manual", "mock", "textract", "external", "doctr"]


class OCRResult(BaseModel):
    metadata: OCRMetadata = Field(default_factory=OCRMetadata)
    handwritten_text: str = ""
    lines: list[OCRLine] = Field(default_factory=list)
    source: OCRSource
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def add_validation_warnings(self):
        if not self.handwritten_text.strip():
            warning = "handwritten_text is empty"
            if warning not in self.warnings:
                self.warnings.append(warning)
        return self
