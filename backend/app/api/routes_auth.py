from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import select

from app.db import get_factory
from app.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class DevUserIn(BaseModel):
    email: str


@router.post("/dev-user")
def dev_user(body: DevUserIn) -> dict[str, str]:
    """Create or fetch a user by email. Dev only; real auth arrives in Phase 5."""
    with get_factory()() as session, session.begin():
        user = session.scalar(select(User).where(User.email == body.email))
        if user is None:
            user = User(email=body.email)
            session.add(user)
            session.flush()
        return {"user_id": str(user.id)}
