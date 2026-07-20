import logging
import threading
import lancedb
from infrastructure.config.settings import settings

logger = logging.getLogger(__name__)


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
                    try:
                        logger.info(f"Initializing LanceDB connection at URI: {self._uri}")
                        # Natively creates local directories or resolves cloud URIs automatically
                        self._connection = lancedb.connect(self._uri)
                        logger.info("LanceDB connection established successfully.")
                    except Exception as ex:
                        logger.error(f"Failed to connect to LanceDB at {self._uri}: {ex}")
                        raise

        return self._connection

    def swap_tables(self, source_table_name: str, target_table_name: str) -> None:
        """
        Promotes a staging/sync table to the active production table.
        Overwrites target production table using source Arrow dataset, then drops source table.

        Args:
            source_table_name (str): Name of the temporary staging table (e.g., 'professors_sync').
            target_table_name (str): Name of the live production table (e.g., 'professors').
        """
        conn = self.get_connection()
        with self._lock:
            tables = conn.list_tables().tables
            if source_table_name in tables:
                logger.info(f"Swapping table '{source_table_name}' -> '{target_table_name}'")
                source_table = conn.open_table(source_table_name)
                arrow_data = source_table.to_arrow()
                conn.create_table(target_table_name, data=arrow_data, mode="overwrite")
                conn.drop_table(source_table_name)
                logger.info(f"Successfully promoted '{source_table_name}' to '{target_table_name}'.")
            else:
                logger.warning(
                    f"Table swap aborted: Source table '{source_table_name}' does not exist."
                )


# Instantiate a centralized client instance to manage connection sharing
lancedb_client = LanceDbClient()