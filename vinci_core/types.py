# types.py

from typing import List, Dict, Union, Optional, Any, Tuple

# Type aliases for commonly used types
Integer = int
Float = float
String = str
Boolean = bool

# A type alias for a mapping of strings to strings (e.g., configuration settings)
Config = Dict[str, String]

# A generic response type for functions that return a data structure
Response[T] = Dict[str, Union[T, List[T]]]

# Optionally, a type alias representing a nullable type
Nullable[T] = Optional[T]

# Example complex types
UserID = String  # An alias for user identifiers
User = Dict[UserID, String]  # A mapping from UserID to username
Coordinates = Tuple[Float, Float]  # A tuple representing (latitude, longitude)

# Add more type hints and aliases as needed for the project
