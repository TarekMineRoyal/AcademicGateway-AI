import threading
import lancedb
from infrastructure.config.settings import settings


class LanceDbClient:
    """
    Manages the lifecycle and initialization of the LanceDB vector database connection.
    Ensures a single, thread-safe connection instance is shared across the repositories.
    """

    def __init__(self) -> None:
        self._uri = settings.LANCE_DB_URI
        self._connection: lancedb.DBConnection | None = None
        self._lock = threading.Lock()  # Guards connection initialization across threads

    def get_connection(self) -> lancedb.DBConnection:
        """
        Retrieves the active LanceDB connection. Lazily initializes the
        database connection path if it does not yet exist.

        Returns:
            lancedb.DBConnection: The active database connection client wrapper.
        """
        if self._connection is None:
            with self._lock:
                # Double-checked lock pattern to prevent simultaneous allocations
                if self._connection is None:
                    # Natively creates local directories or resolves cloud URIs automatically
                    self._connection = lancedb.connect(self._uri)

        return self._connection


# Instantiate a centralized client instance to manage connection sharing
lancedb_client = LanceDbClient()