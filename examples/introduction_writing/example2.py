# HDR user registration validation example
from hdr import BaseModel, verify, quote

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str

    def __init__(self, **data):
        super().__init__(**data)
        verify(f"{quote(self.username)} is 3-20 chars (letters/numbers/_)")
        verify(f"{quote(self.email)} is valid")
        verify(f"{quote(self.password)} is ≥8 chars (upper/lower/number)")

user = UserRegistration(username="john_doe123", email="john@example.com", password="SecurePass123")
print("✅ Valid registration!")