# vinci_core/models/base.py

class BaseModel:
    name: str

    async def generate(self, context: dict) -> str:
        raise NotImplementedError
