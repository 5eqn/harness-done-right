# HDR user registration validation example
import re

from hdr import BaseModel, verify, quote


class UserRegistration(BaseModel):
    username: str
    email: str
    password: str

    def __init__(self, **data):
        super().__init__(**data)
        # Username: 3-20 chars, letters/numbers/underscore
        assert 3 <= len(self.username) <= 20, "Username must be 3-20 chars"
        assert re.match(r"^[a-zA-Z0-9_]+$", self.username), (
            "Username must be letters/numbers/_"
        )
        verify(f"{quote(self.email)} is valid")
        # Password: ≥8 chars with upper, lower, and number
        assert len(self.password) >= 8, "Password must be at least 8 chars"
        assert any(c.isupper() for c in self.password), "Password must have uppercase"
        assert any(c.islower() for c in self.password), "Password must have lowercase"
        assert any(c.isdigit() for c in self.password), "Password must have a digit"


user = UserRegistration(
    username="john_doe123", email="john@example.com", password="SecurePass123"
)
print("✅ Valid registration!")
