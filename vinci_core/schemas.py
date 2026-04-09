from pydantic import BaseModel, Field, EmailStr

# Define your Pydantic models here

class User(BaseModel):
    id: int = Field(..., description="The unique identifier for the user")
    username: str = Field(..., max_length=150, description="The username of the user")
    email: EmailStr = Field(..., description="The email address of the user")
    is_active: bool = Field(default=True, description="Indicates if the user is active")

class Item(BaseModel):
    id: int = Field(..., description="The unique identifier for the item")
    name: str = Field(..., max_length=255, description="The name of the item")
    description: str = Field(..., description="A detailed description of the item")
    owner_id: int = Field(..., description="The ID of the item's owner")

class ResponseModel(BaseModel):
    success: bool = Field(..., description="Indicates if the operation was successful")
    data: dict = Field(..., description="The data returned from the operation")
    message: str = Field(default="Operation completed successfully", description="A message regarding the operation status")

# Add additional models and validations as necessary