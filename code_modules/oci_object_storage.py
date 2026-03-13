"""
Module to interact with OCI Object Storage buckets.

This module provides functionality to:
- Download files from an OCI Object Storage bucket
- Upload files to an OCI Object Storage bucket
"""
import oci
import configparser
import os
import logging
from datetime import datetime
from typing import Generator
import pandas as pd
import uuid
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

# Configure application-level logging
logger = logging.getLogger(__name__)


class OCIObjectStorageClient(object):
    """
    Simple Class for interacting with OCI Object Storage Bucket
    """
    def __init__(self):
        """Initialize OCI client by reading config

        Initializes and returns an OCI Object Storage client and bucket information
        by reading configuration from `config.ini`.

        Returns:
            tuple: A tuple containing:
                - oci.object_storage.ObjectStorageClient: The initialized OCI Object Storage client.
                - dict: A dictionary containing bucket namespace, bucket name, compartment ID, and region.
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        # OCI authentication configuration
        oci_config = {
            "user": config['DEFAULT']['user'],
            "key_file": config['DEFAULT']['key_file'],
            "fingerprint": config['DEFAULT']['fingerprint'],
            "tenancy": config['DEFAULT']['tenancy'],
            "region": config['BUCKET']['region']
        }

        # Initialize OCI Object Storage client
        self.bucket_info = {
            "namespace": config['BUCKET']['namespace'],
            "bucket_name": config['BUCKET']['bucket_name'],
            "compartment_id": config['BUCKET']['compartment_id'],
            "region": config['BUCKET']['region']
        }

        self.client = oci.object_storage.ObjectStorageClient(oci_config)

    def put_file_in_bucket_folder(self, local_file_path, bucket_folder_name):
        with open(local_file_path, "rb") as f:

            object_name = bucket_folder_name + '/' +  local_file_path.split('/')[-1]
            response = self.client.put_object( self.bucket_info['namespace'], self.bucket_info['bucket_name'], object_name, f )
            print(f"file write in bucket folder finished. response status code is {response}")
        return response

    def stream_chunks_to_excel_and_upload(
        self,
        chunk_generator: Generator[pd.DataFrame, None, None],
        local_file_path : str,
        bucket_folder_name: str,
        sheet_name: str = "Results",
    ):
        """
        Write DataFrame chunks to a temporary Excel (.xlsx) file on disk
        using openpyxl's write-only mode, then upload it to OCI Object
        Storage — all without loading the full dataset into memory at once.

        Strategy
        --------
        • openpyxl ``write_only=True`` streams rows directly to disk; it
          never holds more than one chunk worth of rows in RAM.
        • The generator yields small DataFrames (e.g. 500 rows each).
          Column headers are written once from the first chunk; subsequent
          chunks append rows without re-reading the file.
        • Once writing is complete the file is uploaded to OCI via a
          streaming file-handle read (no full-file buffer).
        • A Pre-Authenticated Request (PAR) URL is generated and returned
          so the caller can share a direct download link.

        Args:
            chunk_generator: Generator that yields ``pd.DataFrame`` chunks
                             (from ``OracleADBClient.stream_query_chunks``).
            bucket_folder_name (str): Destination folder prefix inside the
                                      bucket (e.g. ``"excel-exports/"``).
            file_prefix (str): Optional prefix for the generated filename.
            sheet_name (str): Excel worksheet name (default: ``"Results"``).

        Returns:
            str: Pre-Authenticated Request (PAR) URL for the uploaded file.

        Raises:
            Exception: Propagates any OCI or IO errors after cleaning up
                       the local temp file.
        """

        # ── 1. Build a unique local temp path ─────────────────────────────
        object_name = bucket_folder_name + os.path.basename(local_file_path)

        logger.info(f"Streaming chunks → {local_file_path} (write-only mode)")

        try:
            # ── 2. Open workbook in write-only mode ────────────────────────
            #    write_only=True means openpyxl never builds a full in-memory
            #    cell tree — rows are flushed to disk immediately.
            wb = Workbook(write_only=True)
            ws = wb.create_sheet(title=sheet_name)

            total_rows = 0
            header_written = False

            for chunk_df in chunk_generator:
                print("Writing chunk")
                if not header_written:
                    header_row = []
                    for col_name in chunk_df.columns:

                        from openpyxl.cell.cell import WriteOnlyCell
                        c = WriteOnlyCell(ws, value=str(col_name))
                        c.font = Font(bold=True, name="Arial")
                        c.fill = PatternFill("solid", start_color="4472C4")
                        c.font = Font(bold=True, name="Arial", color="FFFFFF")
                        c.alignment = Alignment(horizontal="center")
                        header_row.append(c)
                    ws.append(header_row)
                    header_written = True

                # Append each data row from this chunk
                for record in chunk_df.itertuples(index=False, name=None):
                    ws.append(list(record))

                total_rows += len(chunk_df)
                logger.debug(f"Written {total_rows} rows so far …")

            if not header_written:
                # Generator yielded nothing — write an empty sheet
                ws.append(["No data returned"])

            wb.save(local_file_path)
            logger.info(f"All {total_rows} rows saved to {local_file_path}")

            # ── 3. Upload the file to OCI via streaming file-handle ────────
            file_size = os.path.getsize(local_file_path)
            logger.info(f"Uploading {local_file_path} ({file_size} bytes) → bucket/{object_name}")

            with open(local_file_path, "rb") as f:
                self.client.put_object(
                    namespace_name=self.bucket_info["namespace"],
                    bucket_name=self.bucket_info["bucket_name"],
                    object_name=object_name,
                    put_object_body=f,
                    content_type=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                )

            logger.info("Upload complete")

            # ── 4. Create and return PAR URL ───────────────────────────────
            return True

        except Exception:
            logger.exception("Failed during chunked Excel upload")
            raise

        finally:
            # ── 5. Always clean up the temp file ──────────────────────────
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                logger.info(f"Temp file {local_file_path} removed")

    # ------------------------------------------------------------------ #
    #  PAR (Pre-Authenticated Request) helper                             #
    # ------------------------------------------------------------------ #

    def _create_par_url(self, object_name: str) -> str:
        """
        Create a time-limited Pre-Authenticated Request URL for an object
        so it can be downloaded without OCI credentials.

        The PAR expires after 24 hours by default.

        Args:
            object_name (str): Full object path inside the bucket.

        Returns:
            str: Fully-qualified PAR download URL.
        """
        from datetime import timezone, timedelta

        expiry = datetime.now(timezone.utc) + timedelta(hours=24*90)

        par_details = oci.object_storage.models.CreatePreauthenticatedRequestDetails(
            name=f"par-{uuid.uuid4().hex[:8]}",
            object_name=object_name,
            access_type="ObjectRead",
            time_expires=expiry.isoformat(),
        )

        response = self.client.create_preauthenticated_request(
            namespace_name=self.bucket_info['namespace'],
            bucket_name=self.bucket_info['bucket_name'],
            create_preauthenticated_request_details=par_details,
        )

        region = self.bucket_info['region']
        access_uri = response.data.access_uri

        # Build the full URL: https://objectstorage.<region>.oraclecloud.com<access_uri>
        par_url = f"https://objectstorage.{region}.oraclecloud.com{access_uri}"
        logger.info(f"PAR URL created: {par_url}")
        return par_url
