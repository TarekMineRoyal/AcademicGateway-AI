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

    def swap_tables(self, source_table_name: str, target_table_name: str) -> None:
        """
        Promotes a staging/sync table to the active production table.
        Drops the target production table if it exists and renames the source table to target.

        Args:
            source_table_name (str): Name of the temporary staging table (e.g., 'professors_sync').
            target_table_name (str): Name of the live production table (e.g., 'professors').
        """
        conn = self.get_connection()
        with self._lock:
            existing_tables = conn.table_names()
            if target_table_name in existing_tables:
                conn.drop_table(target_table_name)
            conn.rename_table(source_table_name, target_table_name)


# Instantiate a centralized client instance to manage connection sharing
lancedb_client = LanceDbClient()