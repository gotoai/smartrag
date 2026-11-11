"""Extraction: turning a source file into text the chunker can split.

v1 flattens every format to text before chunking (design memo §3.4). Tabular files —
CSV, TSV and Excel — become TSV table blocks (`smartrag.extract.tabular`).
"""

from smartrag.extract.tabular import (
    TABULAR_EXTENSIONS,
    TableBlock,
    TabularDocument,
    convert_tabular,
    is_tabular,
)

__all__ = ["TABULAR_EXTENSIONS", "TableBlock", "TabularDocument", "convert_tabular",
           "is_tabular"]
