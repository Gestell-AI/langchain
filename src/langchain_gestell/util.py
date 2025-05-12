from typing import Optional
from uuid import UUID


def validate_collection_id(internal_coll_id: str, collection_id: Optional[str]) -> str:
    """
    Returns collection_id if it's a valid UUID, otherwise falls back to self.collection_id.
    """
    try:
        if collection_id:
            UUID(collection_id)
            return collection_id
    except ValueError:
        pass

    return internal_coll_id
