"""JWT-based IdentityProvider — stands in for Entra ID during training."""

from datetime import datetime, timedelta, timezone

import jwt

import archive
from loda.core.auth import Login, NotSignedIn, User

TRAINING_SECRET = "loda-training-only-never-use-in-production"
AUDIENCE = "loda-chatbot"
TOKEN_LIFETIME = timedelta(hours=1)
ALREADY_EXPIRED = timedelta(seconds=-1)

IDENTITY_PARAMETERS = ("requested_by", "uploaded_by", "raised_by")


class IdentityProvider:
    """Signs a token at sign-in and verifies it on every request."""

    def __init__(self, secret: str = TRAINING_SECRET):
        self._secret = secret

    def sign_in(self, user_id: str, valid_for: timedelta = TOKEN_LIFETIME) -> Login:
        record = archive.find_user(user_id)
        if record is None:
            raise NotSignedIn(f"There is no user called {user_id}.")
        user = User(
            record["user_id"],
            record["name"],
            record["email"],
            tuple(record["groups"]),
        )
        now = datetime.now(timezone.utc)
        claims = {
            "sub": user.user_id,
            "name": user.name,
            "email": user.email,
            "groups": list(user.groups),
            "aud": AUDIENCE,
            "iat": now,
            "exp": now + valid_for,
        }
        return Login(user, jwt.encode(claims, self._secret, algorithm="HS256"))

    def verify(self, token: str) -> User:
        try:
            claims = jwt.decode(
                token, self._secret, algorithms=["HS256"], audience=AUDIENCE
            )
        except jwt.ExpiredSignatureError as error:
            raise NotSignedIn("Your sign-in has expired. Please sign in again.") from error
        except jwt.InvalidTokenError as error:
            raise NotSignedIn("This sign-in is not valid.") from error
        return User(
            claims["sub"], claims["name"], claims["email"], tuple(claims["groups"])
        )
